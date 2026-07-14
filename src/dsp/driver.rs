// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Smoothing, parameter-target latching, and the driven kernel wrapper.
//!
//! [`SmootherBank`] stores parameter state. The block driver splits blocks at
//! control-rate grid boundaries only and applies sample-stamped event targets
//! at the next boundary. [`KernelProcessor`] owns the smoother
//! bank for a kernel.

#[cfg(debug_assertions)]
use crate::block::Io;
use crate::block::{AudioBlockMut, IoMode};
use crate::context::{ProcessContext, Produced, SubBlock, Tail};
use crate::dsp::sanitize::finite_or_zero;
use crate::error::DspError;
use crate::param::{ParamEvent, ParamId, ParamInfo, ParamSetError, Params, Smoothing};
use crate::processor::Sample;
use crate::spec::ProcessSpec;
use crate::traits::{Kernel, Processor};
use core::marker::PhantomData;

/// Control-rate period in frames. The smoother bank advances one step per
/// `CR_STEP` absolute frames.
const CR_STEP: u64 = 32;

/// The longest run [`drive`] ever hands a kernel: one control-rate cell.
///
/// `drive` splits blocks at grid boundaries, so a `SubBlock` spans at most this
/// many frames. `SubBlock` construction is crate-internal, which makes the
/// bound load-bearing. Kernels may size per-run scratch to it.
pub(crate) const MAX_RUN_FRAMES: usize = CR_STEP as usize;

// ---------------------------------------------------------------------------
// The smoother bank
// ---------------------------------------------------------------------------

/// One parameter's smoother. Ramps `cur` toward `target` as a raw value at the
/// control rate.
#[derive(Debug)]
struct Smoother {
    id: ParamId,
    cur: f64,
    target: f64,
    range: (f64, f64),
    default: f64,
    steps: f64,
    updates_remaining: u64,
    step_delta: f64,
    shape: Smoothing,
    /// Shape-dependent coefficient. This is the fixed `OnePole` approach
    /// coefficient or the current `Exponential` target's per-step ratio.
    /// It is unused by `Step` and `Linear`.
    coeff: f64,
}

/// One smoother per declared parameter.
///
/// The bank is built in `prepare` and stepped at the control rate by the block
/// drivers.
#[derive(Debug, Default)]
pub struct SmootherBank {
    smoothers: Vec<Smoother>,
}

impl SmootherBank {
    /// Build a bank for validated declared parameters, sized to `spec`.
    ///
    /// Each smoother derives its step size from its own
    /// `ParamInfo::smoothing_ms`, so ramp speed is per-parameter. The exact
    /// meaning of `smoothing_ms` is shape-dependent; see
    /// [`Smoothing`](crate::parameter::Smoothing) for the per-variant math.
    fn new(params: &[ParamInfo], spec: &ProcessSpec) -> Self {
        let smoothers = params
            .iter()
            .map(|p| {
                let smoothing_ms = if p.smoothing_ms.is_finite() && p.smoothing_ms > 0.0 {
                    p.smoothing_ms
                } else {
                    5.0
                };
                let steps =
                    ((smoothing_ms * 1e-3 * f64::from(spec.sample_rate)) / CR_STEP as f64).max(1.0);
                let default = if p.default.is_finite()
                    && p.range.0.is_finite()
                    && p.range.1.is_finite()
                    && p.range.0 <= p.range.1
                {
                    p.default.clamp(p.range.0, p.range.1)
                } else {
                    0.0
                };
                // OnePole has a fixed coefficient. Exponential derives its
                // ratio when each new target is latched.
                let coeff = match p.smoothing {
                    Smoothing::OnePole => 1.0 - super::math::exp(-1.0 / steps),
                    _ => 1.0,
                };
                Smoother {
                    id: p.id,
                    cur: default,
                    target: default,
                    range: p.range,
                    default,
                    steps,
                    updates_remaining: 0,
                    step_delta: 0.0,
                    shape: p.smoothing,
                    coeff,
                }
            })
            .collect();
        Self { smoothers }
    }

    /// Validate the declared parameters and build a bank sized to `spec`.
    ///
    /// # Errors
    /// Returns [`DspError::InvalidParam`] when ids, ranges, defaults, smoothing,
    /// or normalized value scaling violate the parameter metadata contract.
    /// Returns [`DspError::OverBudget`] when the bank does not fit within
    /// `spec.max_memory`.
    pub fn try_new(params: &[ParamInfo], spec: &ProcessSpec) -> Result<Self, DspError> {
        validate_param_info(params)?;
        crate::dsp::memory::MemoryLayout::new()
            .array::<Smoother>(params.len())
            .preflight(spec.max_memory)?;
        Ok(Self::new(params, spec))
    }

    /// State bytes, counted into the owning processor's footprint.
    #[must_use]
    pub fn footprint(&self) -> usize {
        Self::footprint_for(self.smoothers.len())
    }

    /// The bytes a bank of `param_count` smoothers occupies, computable
    /// before construction so budget checks can run without committing state.
    #[must_use]
    pub fn footprint_for(param_count: usize) -> usize {
        param_count.saturating_mul(std::mem::size_of::<Smoother>())
    }

    /// Set a parameter's target, clamped to its declared range.
    ///
    /// # Errors
    /// Returns [`ParamSetError::UnknownParam`] if `id` was not declared, or
    /// [`ParamSetError::NonFiniteValue`] if `value` is NaN or infinite.
    pub fn set_target(&mut self, id: ParamId, value: f64) -> Result<(), ParamSetError> {
        if !value.is_finite() {
            return Err(ParamSetError::NonFiniteValue { param: id, value });
        }
        if let Some(s) = self.smoothers.iter_mut().find(|s| s.id == id) {
            let target = value.clamp(s.range.0, s.range.1);
            if target == s.target {
                return Ok(());
            }
            s.target = target;
            match s.shape {
                Smoothing::Linear => {
                    s.updates_remaining = s.steps.ceil() as u64;
                    s.step_delta = (s.target - s.cur).abs() / s.steps;
                }
                Smoothing::Exponential => {
                    s.updates_remaining = s.steps.ceil() as u64;
                    // Exponential metadata has a positive range. Computing
                    // the ratio through logarithms also handles ranges whose
                    // direct quotient would overflow.
                    s.coeff = if s.cur > 0.0 && s.target > 0.0 {
                        let log_step =
                            (super::math::ln(s.target) - super::math::ln(s.cur)).abs() / s.steps;
                        super::math::exp(log_step)
                    } else {
                        1.0
                    };
                }
                Smoothing::Step | Smoothing::OnePole => {}
            }
            Ok(())
        } else {
            Err(ParamSetError::UnknownParam(id))
        }
    }

    /// Set a parameter's current value and target to `value` immediately,
    /// clamped to its declared range, with no ramp.
    ///
    /// This is lower-level smoother-bank vocabulary. Hosts normally call
    /// [`Processor::set_parameter_immediate`](crate::processor::Processor::set_parameter_immediate),
    /// which delegates here; smoothing only applies to event targets.
    ///
    /// # Errors
    /// Returns [`ParamSetError::UnknownParam`] if `id` was not declared, or
    /// [`ParamSetError::NonFiniteValue`] if `value` is NaN or infinite.
    pub fn set_immediate(&mut self, id: ParamId, value: f64) -> Result<(), ParamSetError> {
        if !value.is_finite() {
            return Err(ParamSetError::NonFiniteValue { param: id, value });
        }
        if let Some(s) = self.smoothers.iter_mut().find(|s| s.id == id) {
            let v = value.clamp(s.range.0, s.range.1);
            s.cur = v;
            s.target = v;
            s.updates_remaining = 0;
            Ok(())
        } else {
            Err(ParamSetError::UnknownParam(id))
        }
    }

    /// Advance every smoother one control-rate step.
    pub fn step(&mut self) {
        for s in &mut self.smoothers {
            match s.shape {
                Smoothing::Step => s.cur = s.target,
                Smoothing::Linear => {
                    if s.cur == s.target {
                        s.updates_remaining = 0;
                    } else if s.updates_remaining <= 1 {
                        s.cur = s.target;
                        s.updates_remaining = 0;
                    } else {
                        let d = s.target - s.cur;
                        s.cur += s.step_delta.copysign(d);
                        s.updates_remaining -= 1;
                    }
                }
                Smoothing::OnePole => s.cur += (s.target - s.cur) * s.coeff,
                // Constant ratio for the current target in each direction;
                // snap on arrival.
                Smoothing::Exponential => {
                    if s.target == s.cur {
                        s.updates_remaining = 0;
                    } else if s.updates_remaining <= 1 {
                        s.cur = s.target;
                        s.updates_remaining = 0;
                    } else if s.target > s.cur {
                        let next = s.cur * s.coeff;
                        s.cur = next.min(s.target);
                        s.updates_remaining -= 1;
                    } else if s.target < s.cur {
                        let next = s.cur / s.coeff;
                        s.cur = next.max(s.target);
                        s.updates_remaining -= 1;
                    }
                }
            }
            s.cur = finite_or_zero(s.cur).clamp(s.range.0, s.range.1);
        }
    }

    /// The current smoothed value for `id`.
    #[must_use]
    pub fn value(&self, id: ParamId) -> Option<f64> {
        self.smoothers.iter().find(|s| s.id == id).map(|s| s.cur)
    }

    /// The current smoothed value at declaration `index`.
    ///
    /// Used by [`Params::from_bank`] loaders; ids are validated sequential in
    /// `prepare`, so the index is the id.
    #[must_use]
    pub fn value_at(&self, index: usize) -> Option<f64> {
        self.smoothers.get(index).map(|s| s.cur)
    }

    /// Return every smoother to its default.
    pub fn reset(&mut self) {
        for s in &mut self.smoothers {
            s.cur = s.default;
            s.target = s.default;
            s.updates_remaining = 0;
        }
    }
}

fn validate_param_info(params: &[ParamInfo]) -> Result<(), DspError> {
    for (index, param) in params.iter().enumerate() {
        if param.id.0 as usize != index {
            return Err(DspError::InvalidParam(
                "parameter ids must be sequential from 0 in declaration order",
            ));
        }
        if !param.range.0.is_finite() || !param.range.1.is_finite() || param.range.0 > param.range.1
        {
            return Err(DspError::InvalidParam(
                "parameter ranges must be finite and ordered",
            ));
        }
        if !param.default.is_finite() {
            return Err(DspError::InvalidParam("parameter defaults must be finite"));
        }
        if param.default < param.range.0 || param.default > param.range.1 {
            return Err(DspError::InvalidParam(
                "parameter defaults must be inside their range",
            ));
        }
        if !param.smoothing_ms.is_finite() || param.smoothing_ms <= 0.0 {
            return Err(DspError::InvalidParam(
                "parameter smoothing_ms must be finite and positive",
            ));
        }
        if param.smoothing == Smoothing::Exponential && param.range.0 <= 0.0 {
            return Err(DspError::InvalidParam(
                "Exponential smoothing requires a positive range minimum",
            ));
        }
        if param.value_scale == crate::parameter::ValueScale::Logarithmic && param.range.0 <= 0.0 {
            return Err(DspError::InvalidParam(
                "Logarithmic value scale requires a positive range minimum",
            ));
        }
    }
    Ok(())
}

fn debug_assert_events_valid(events: &[ParamEvent], frames: usize) {
    debug_assert!(
        events.iter().all(|e| (e.offset as usize) < frames),
        "parameter events must have offset < frames"
    );
    debug_assert!(
        events
            .windows(2)
            .all(|pair| pair[0].offset <= pair[1].offset),
        "parameter events must be sorted by offset"
    );
    debug_assert!(
        events.iter().all(|e| e.value.is_finite()),
        "parameter event values must be finite"
    );
}

/// Latch targets from `events[*ev..]` stamped at or before `upto`, advancing
/// `*ev` past everything consumed. Invalid entries (offset at or past the
/// block, non-finite values) are skipped deterministically.
fn latch_events(
    bank: &mut SmootherBank,
    events: &[ParamEvent],
    ev: &mut usize,
    upto: usize,
    frames: usize,
) {
    while let Some(event) = events.get(*ev) {
        let offset = event.offset as usize;
        if offset > upto {
            break;
        }
        *ev += 1;
        if offset < frames && event.value.is_finite() {
            let _ = bank.set_target(event.param, event.value);
        }
    }
}

// ---------------------------------------------------------------------------
// The drivers
// ---------------------------------------------------------------------------

/// Drive a [`Kernel`] over one block.
///
/// The block is split at control-rate grid boundaries anchored to the absolute
/// sample timeline. Every target stamped at or before a boundary has latched
/// before that boundary's step, so targets are quantized to the first boundary
/// at or after their timestamp. Stream start (`sample_pos == 0`) counts as a
/// boundary: offset-0 events latch and step at frame 0. Targets stamped after
/// a block's last boundary still latch before the block ends and carry into the
/// next boundary step.
pub(crate) fn drive<T: Sample, K: Kernel<T>>(
    k: &mut K,
    bank: &mut SmootherBank,
    ctx: &mut ProcessContext<'_, '_, T>,
) {
    let frames = ctx.frames;
    debug_assert_events_valid(ctx.events, frames);
    if frames == 0 {
        return;
    }
    let pos = ctx.sample_pos;
    let events = ctx.events;
    let sidechain = ctx.sidechain;
    let mut start = 0usize;
    let mut ev = 0usize;
    // The next grid cell whose boundary step is still pending. The stream-start
    // boundary has not stepped yet; for a later block, every boundary at or
    // before the previous frame has.
    let mut pending_cell = if pos == 0 { 0 } else { (pos - 1) / CR_STEP + 1 };

    while start < frames {
        let abs = pos + start as u64;
        // Latch every target stamped at or before this run's first frame.
        latch_events(bank, events, &mut ev, start, frames);
        // Step once per boundary crossed into (stream start included).
        let cell = abs / CR_STEP;
        while pending_cell <= cell {
            bank.step();
            pending_cell += 1;
        }
        // Render to the next grid boundary or the block end.
        let next_grid = abs - (abs % CR_STEP) + CR_STEP;
        let end = ((next_grid - pos) as usize).min(frames);
        let len = end - start;
        // The bound kernels size fixed per-run scratch against: this driver
        // is the party guaranteeing it, so it asserts it at the source.
        debug_assert!(len <= MAX_RUN_FRAMES, "run length is within one CR cell");
        // Load the typed parameter snapshot for this fixed-parameter run.
        let params = K::Params::from_bank(bank);
        {
            let mut sub = SubBlock {
                io: &mut ctx.main,
                sc: sidechain,
                start,
                len,
            };
            k.render(&mut sub, &params);
        }
        start = end;
    }
    // Latch targets stamped after the last boundary so the next block's first
    // step sees them.
    latch_events(bank, events, &mut ev, frames, frames);
}

// ---------------------------------------------------------------------------
// Debug-time host-geometry validation
// ---------------------------------------------------------------------------

/// The minimal prepared contract retained for debug-time geometry validation.
///
/// Inline scalar state: excluded from `memory_footprint()`, which counts owned
/// buffer payload bytes only.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(debug_assertions), allow(dead_code))]
pub(crate) struct PreparedContract {
    pub(crate) max_block: usize,
    pub(crate) channels: usize,
    pub(crate) io_mode: IoMode,
    pub(crate) sidechain_inputs: usize,
}

/// Debug-assert that a block honors the prepared host contract before it
/// reaches kernel indexing.
///
/// Host geometry (channel count, block size, I/O shape, sidechain buses) is a
/// host precondition, not a recoverable error: violations panic in debug builds
/// with a specific message. Release behavior is not specified and may produce
/// incorrect output or panic deeper in the kernel. Release processing stays
/// branch-free.
///
/// Shared by [`KernelProcessor`] and the crate's direct [`Processor`]
/// implementations.
#[inline]
pub(crate) fn debug_validate_geometry<T: Sample>(
    prepared: Option<&PreparedContract>,
    ctx: &ProcessContext<'_, '_, T>,
) {
    #[cfg(debug_assertions)]
    {
        let p = prepared.expect("process requires a successful prepare");
        assert!(
            ctx.frames <= p.max_block,
            "block frames ({}) must not exceed the prepared max_block ({})",
            ctx.frames,
            p.max_block
        );
        match (&ctx.main, p.io_mode) {
            (Io::InPlace(block), IoMode::InPlace) => {
                assert!(
                    block.frames() == ctx.frames,
                    "in-place main frame count ({}) must equal ctx.frames ({})",
                    block.frames(),
                    ctx.frames
                );
                assert!(
                    block.channels() == p.channels,
                    "main channel count ({}) must equal the prepared channel count ({})",
                    block.channels(),
                    p.channels
                );
            }
            (Io::OutputOnly(output), IoMode::OutputOnly) => {
                assert!(
                    output.frames() == ctx.frames,
                    "output-only main frame count ({}) must equal ctx.frames ({})",
                    output.frames(),
                    ctx.frames
                );
                assert!(
                    output.channels() == p.channels,
                    "main channel count ({}) must equal the prepared channel count ({})",
                    output.channels(),
                    p.channels
                );
            }
            (Io::Split { input, output }, IoMode::Split) => {
                assert!(
                    input.frames() == ctx.frames,
                    "split input frame count ({}) must equal ctx.frames ({})",
                    input.frames(),
                    ctx.frames
                );
                assert!(
                    output.frames() == ctx.frames,
                    "split output frame count ({}) must equal ctx.frames ({})",
                    output.frames(),
                    ctx.frames
                );
                assert!(
                    output.channels() == p.channels,
                    "main channel count ({}) must equal the prepared channel count ({})",
                    output.channels(),
                    p.channels
                );
                assert!(
                    input.channels() == output.channels(),
                    "split input and output must have equal channel counts"
                );
                assert!(
                    input.frames() == output.frames(),
                    "split input and output must have equal frame counts"
                );
            }
            _ => panic!("ctx.main I/O shape must match the processor's io_mode()"),
        }
        assert!(
            ctx.sidechain.len() == p.sidechain_inputs,
            "sidechain bus count ({}) must equal sidechain_inputs() ({})",
            ctx.sidechain.len(),
            p.sidechain_inputs
        );
        for (bus, sc) in ctx.sidechain.iter().enumerate() {
            assert!(
                sc.frames() >= ctx.frames,
                "sidechain bus {bus} has {} frames, fewer than the block's {}",
                sc.frames(),
                ctx.frames
            );
        }
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = (prepared, ctx);
    }
}

/// Debug-assert that a flush output block uses the prepared main-channel
/// geometry. Flush capacity is per call and is intentionally independent of
/// `ProcessSpec::max_block`, which constrains input processing only.
#[inline]
pub(crate) fn debug_validate_flush_geometry<T: Sample>(
    prepared: Option<&PreparedContract>,
    out: &AudioBlockMut<'_, '_, T>,
) {
    #[cfg(debug_assertions)]
    {
        let p = prepared.expect("flush requires a successful prepare");
        assert!(
            out.channels() == p.channels,
            "flush output channel count ({}) must equal the prepared channel count ({})",
            out.channels(),
            p.channels
        );
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = (prepared, out);
    }
}

// ---------------------------------------------------------------------------
// The wrapper that owns the bank
// ---------------------------------------------------------------------------

/// Owns a kernel plus the framework smoother bank and drives it as a
/// [`Processor`].
///
/// `prepare` builds the smoother bank from `param_info()`. `process` runs the
/// internal block driver. Metadata and tail handling delegate to the kernel.
/// The sample type is part of the wrapper type, defaulting to f32 through
/// [`KernelProcessor::new`]. Use [`KernelProcessor::with_sample_type`] for f64
/// or another explicit sample type.
///
/// Debug builds validate each block against the prepared contract (frame
/// count, channel count, I/O shape, and sidechain buses) before it reaches the
/// kernel; release builds do not pay for those checks.
///
/// When [`ProcessSpec::max_memory`] is set, `prepare` reserves the smoother
/// bank's footprint first and hands the kernel only the remainder as its
/// sub-budget. Built-in kernels preflight their known layouts before
/// allocating; a downstream kernel is responsible for honoring its sub-budget
/// the same way, because a kernel that allocates first is only rejected by
/// the wrapper's post-prepare total check, after the allocation happened.
#[derive(Debug)]
pub struct KernelProcessor<K, T: Sample = f32> {
    kernel: K,
    bank: SmootherBank,
    prepared: Option<PreparedContract>,
    _sample: PhantomData<fn() -> T>,
}

impl<K> KernelProcessor<K, f32> {
    /// Wrap a kernel for f32 processing. The smoother bank is built in
    /// `prepare`.
    #[must_use]
    pub fn new(kernel: K) -> Self {
        Self::with_sample_type(kernel)
    }
}

impl<K, T: Sample> KernelProcessor<K, T> {
    /// Wrap a kernel for an explicit sample type. The smoother bank is built in
    /// `prepare`.
    #[must_use]
    pub fn with_sample_type(kernel: K) -> Self {
        Self {
            kernel,
            bank: SmootherBank::default(),
            prepared: None,
            _sample: PhantomData,
        }
    }
}

impl<T: Sample, K: Kernel<T>> Processor<T> for KernelProcessor<K, T> {
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        // `param_info` is constructor-set, so the bank can be budgeted before
        // allocation. Wrapper state is transactional: the bank is installed only
        // after budget checks pass. A kernel that ignores its sub-budget may
        // still allocate before the total check rejects it; failed prepares
        // leave the processor unprepared.
        self.prepared = None;
        // The typed parameter struct and the metadata list must agree in
        // length: extra typed fields would panic in `Params::from_bank` during
        // render, and extra metadata would expose a parameter the kernel never
        // reads.
        if K::Params::COUNT != self.kernel.param_info().len() {
            return Err(DspError::InvalidParam(
                "typed parameter count must match param_info",
            ));
        }
        validate_param_info(self.kernel.param_info())?;
        let bank_bytes = crate::dsp::memory::MemoryLayout::new()
            .array::<Smoother>(self.kernel.param_info().len())
            .preflight(spec.max_memory)?;
        let kernel_spec = if let Some(cap) = spec.max_memory {
            // Hand the kernel only what remains after the bank.
            let remaining = cap.checked_sub(bank_bytes).ok_or(DspError::OverBudget {
                needed: bank_bytes,
                cap,
            })?;
            ProcessSpec {
                max_memory: Some(remaining),
                ..spec
            }
        } else {
            spec
        };
        self.kernel.prepare(kernel_spec)?;
        // Kernels are not required to honor `max_memory` themselves, so keep a
        // belt-and-braces total check after the kernel has prepared.
        if let Some(cap) = spec.max_memory {
            let total = self
                .kernel
                .memory_footprint()
                .checked_add(bank_bytes)
                .ok_or(DspError::OverBudget {
                    needed: usize::MAX,
                    cap,
                })?;
            if total > cap {
                return Err(DspError::OverBudget { needed: total, cap });
            }
        }
        self.bank = SmootherBank::new(self.kernel.param_info(), &spec);
        self.prepared = Some(PreparedContract {
            max_block: spec.max_block,
            channels: spec.channels,
            io_mode: self.kernel.io_mode(),
            sidechain_inputs: self.kernel.sidechain_inputs(),
        });
        Ok(())
    }
    fn reset(&mut self) {
        self.kernel.reset();
        self.bank.reset();
    }
    fn latency(&self) -> usize {
        self.kernel.latency()
    }
    fn tail(&self) -> Tail {
        self.kernel.tail()
    }
    fn io_mode(&self) -> IoMode {
        self.kernel.io_mode()
    }
    fn memory_footprint(&self) -> usize {
        self.kernel
            .memory_footprint()
            .saturating_add(self.bank.footprint())
    }
    fn param_info(&self) -> &[ParamInfo] {
        self.kernel.param_info()
    }
    fn sidechain_inputs(&self) -> usize {
        self.kernel.sidechain_inputs()
    }
    fn set_parameter_immediate(&mut self, id: ParamId, value: f64) -> Result<(), ParamSetError> {
        self.bank.set_immediate(id, value)
    }
    fn process(&mut self, ctx: &mut ProcessContext<'_, '_, T>) {
        debug_validate_geometry(self.prepared.as_ref(), ctx);
        drive(&mut self.kernel, &mut self.bank, ctx);
    }
    fn flush(&mut self, out: &mut AudioBlockMut<'_, '_, T>) -> Produced {
        debug_validate_flush_geometry(self.prepared.as_ref(), out);
        self.kernel.flush(out)
    }
}

// ---------------------------------------------------------------------------
// Tests for smoother behavior, driver splitting, and wrapper boundaries.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::param::{ParamEvent, Unit};

    /// A recorded render trajectory: one `(start, len, value)` per render call.
    type Trace = Vec<(usize, usize, f64)>;

    fn spec_cap(max_memory: Option<usize>) -> ProcessSpec {
        ProcessSpec {
            sample_rate: 48_000,
            channels: 1,
            max_block: 128,
            max_memory,
        }
    }
    fn spec() -> ProcessSpec {
        spec_cap(None)
    }
    fn pinfo(id: ParamId, range: (f64, f64), default: f64, smoothing: Smoothing) -> ParamInfo {
        ParamInfo::new(id, "p", range, default, Unit::Linear).with_smoothing(smoothing)
    }
    fn ev(offset: u32, param: ParamId, value: f64) -> ParamEvent {
        ParamEvent {
            offset,
            param,
            value,
        }
    }
    fn set_target(bank: &mut SmootherBank, id: ParamId, value: f64) {
        bank.set_target(id, value).expect("known parameter id");
    }
    fn set_immediate(bank: &mut SmootherBank, id: ParamId, value: f64) {
        bank.set_immediate(id, value).expect("known parameter id");
    }
    fn value(bank: &SmootherBank, id: ParamId) -> f64 {
        bank.value(id).expect("known parameter id")
    }
    fn value_at(bank: &SmootherBank, index: usize) -> f64 {
        bank.value_at(index).expect("known parameter index")
    }

    // SmootherBank behavior.

    #[test]
    fn footprint_counts_every_smoother() {
        let params: Vec<ParamInfo> = (0..3)
            .map(|i| pinfo(ParamId(i), (0.0, 1.0), 0.0, Smoothing::Step))
            .collect();
        let bank = SmootherBank::new(&params, &spec());
        assert_eq!(bank.footprint(), 3 * std::mem::size_of::<Smoother>());
    }

    #[test]
    fn public_construction_rejects_invalid_metadata() {
        let invalid = pinfo(ParamId(0), (1.0, 0.0), 0.5, Smoothing::Linear);
        assert!(matches!(
            SmootherBank::try_new(&[invalid], &spec()),
            Err(DspError::InvalidParam(_))
        ));

        for range in [(f64::NAN, 1.0), (0.0, f64::INFINITY)] {
            let invalid = pinfo(ParamId(0), range, 0.5, Smoothing::Linear);
            assert!(matches!(
                SmootherBank::try_new(&[invalid], &spec()),
                Err(DspError::InvalidParam(_))
            ));
        }

        let point = pinfo(ParamId(0), (0.5, 0.5), 0.5, Smoothing::Linear);
        SmootherBank::try_new(&[point], &spec()).expect("a single-value range is valid");
    }

    #[test]
    fn public_construction_checks_the_bank_budget_before_allocation() {
        let params = [pinfo(ParamId(0), (0.0, 1.0), 0.5, Smoothing::Linear)];
        assert!(matches!(
            SmootherBank::try_new(&params, &spec_cap(Some(0))),
            Err(DspError::OverBudget { .. })
        ));
    }

    #[test]
    fn value_returns_the_right_smoother_or_none() {
        // Distinct smoother values verify lookup by id.
        let params = vec![
            pinfo(ParamId(1), (0.0, 10.0), 0.0, Smoothing::Step),
            pinfo(ParamId(2), (0.0, 10.0), 0.0, Smoothing::Step),
        ];
        let mut bank = SmootherBank::new(&params, &spec());
        set_target(&mut bank, ParamId(1), 3.0);
        set_target(&mut bank, ParamId(2), 7.0);
        bank.step();
        assert_eq!(value(&bank, ParamId(1)), 3.0);
        assert_eq!(value(&bank, ParamId(2)), 7.0);
        assert!(bank.value(ParamId(99)).is_none(), "unknown id is None");
    }

    #[test]
    fn value_at_reads_by_declaration_index() {
        // `value_at` is the by-index readout used by `Params::from_bank`.
        let params = vec![
            pinfo(ParamId(0), (0.0, 10.0), 0.0, Smoothing::Step),
            pinfo(ParamId(1), (0.0, 10.0), 0.0, Smoothing::Step),
        ];
        let mut bank = SmootherBank::new(&params, &spec());
        set_target(&mut bank, ParamId(0), 3.0);
        set_target(&mut bank, ParamId(1), 7.0);
        bank.step();
        assert_eq!(bank.value_at(0), bank.value(ParamId(0)));
        assert_eq!(bank.value_at(1), bank.value(ParamId(1)));
        assert_eq!(value_at(&bank, 0), 3.0);
        assert_eq!(value_at(&bank, 1), 7.0);
        assert!(bank.value_at(99).is_none(), "out-of-range index is None");
    }

    #[test]
    fn set_target_clamps_to_the_declared_range() {
        let mut bank = SmootherBank::new(
            &[pinfo(ParamId(0), (-2.0, 3.0), 0.0, Smoothing::Step)],
            &spec(),
        );
        set_target(&mut bank, ParamId(0), 100.0);
        bank.step();
        assert_eq!(value(&bank, ParamId(0)), 3.0, "above range clamps to max");
        set_target(&mut bank, ParamId(0), -100.0);
        bank.step();
        assert_eq!(value(&bank, ParamId(0)), -2.0, "below range clamps to min");
    }

    #[test]
    fn step_shapes_jump_or_ramp() {
        // Step reaches target in one step. Linear moves over multiple steps.
        let mut step = SmootherBank::new(
            &[pinfo(ParamId(0), (0.0, 10.0), 0.0, Smoothing::Step)],
            &spec(),
        );
        set_target(&mut step, ParamId(0), 8.0);
        step.step();
        assert_eq!(value(&step, ParamId(0)), 8.0);

        let mut lin = SmootherBank::new(
            &[pinfo(ParamId(0), (0.0, 10.0), 0.0, Smoothing::Linear)],
            &spec(),
        );
        set_target(&mut lin, ParamId(0), 10.0);
        lin.step();
        let one = value(&lin, ParamId(0));
        assert!(one > 0.0 && one < 10.0, "one Linear step is partial: {one}");
        for _ in 0..200 {
            lin.step();
        }
        assert_eq!(
            value(&lin, ParamId(0)),
            10.0,
            "Linear settles exactly at target"
        );

        // A lower target moves the current downward.
        let mut down = SmootherBank::new(
            &[pinfo(ParamId(0), (0.0, 10.0), 10.0, Smoothing::Linear)],
            &spec(),
        );
        set_target(&mut down, ParamId(0), 0.0);
        down.step();
        assert!(
            value(&down, ParamId(0)) < 10.0,
            "Linear moves toward a lower target"
        );
    }

    #[test]
    fn step_ignores_smoothing_ms() {
        // `Step` jumps at the next boundary no matter what smoothing time the
        // metadata declares. The declared value is metadata-only.
        for ms in [5.0, 5_000.0] {
            let mut bank = SmootherBank::new(
                &[pinfo_ramp(
                    ParamId(0),
                    (0.0, 10.0),
                    0.0,
                    Smoothing::Step,
                    ms,
                )],
                &spec(),
            );
            set_target(&mut bank, ParamId(0), 8.0);
            bank.step();
            assert_eq!(
                value(&bank, ParamId(0)),
                8.0,
                "Step with smoothing_ms = {ms} still jumps in one step"
            );
        }
    }

    #[test]
    fn onepole_follows_the_exact_recurrence() {
        // Pin the first two OnePole steps to the recurrence.
        let mut bank = SmootherBank::new(
            &[pinfo(ParamId(0), (0.0, 1.0), 0.0, Smoothing::OnePole)],
            &spec(),
        );
        // Recompute the coefficient used by `new`.
        let steps = ((5.0e-3 * 48_000.0) / CR_STEP as f64).max(1.0);
        let coeff = 1.0 - crate::dsp::math::exp(-1.0 / steps);
        set_target(&mut bank, ParamId(0), 1.0);
        bank.step();
        let c1 = coeff; // from 0: 0 + (1 - 0) * coeff
        assert!(
            (value(&bank, ParamId(0)) - c1).abs() < 1e-15,
            "first OnePole step"
        );
        bank.step();
        let c2 = c1 + (1.0 - c1) * coeff;
        assert!(
            (value(&bank, ParamId(0)) - c2).abs() < 1e-15,
            "second OnePole step"
        );
        // The approach is monotone and bounded.
        let mut prev = value(&bank, ParamId(0));
        for _ in 0..100 {
            bank.step();
            let v = value(&bank, ParamId(0));
            assert!(v >= prev && v <= 1.0, "OnePole is monotone and bounded");
            prev = v;
        }
    }

    #[test]
    fn reset_restores_every_default() {
        let mut bank = SmootherBank::new(
            &[pinfo(ParamId(0), (0.0, 10.0), 4.0, Smoothing::Step)],
            &spec(),
        );
        set_target(&mut bank, ParamId(0), 9.0);
        bank.step();
        assert_eq!(value(&bank, ParamId(0)), 9.0);
        bank.reset();
        assert_eq!(
            value(&bank, ParamId(0)),
            4.0,
            "reset returns to the default"
        );
    }

    #[test]
    fn set_immediate_snaps_current_and_target_without_a_ramp() {
        let mut bank = SmootherBank::new(
            &[pinfo(ParamId(0), (0.0, 10.0), 0.0, Smoothing::Linear)],
            &spec(),
        );
        set_immediate(&mut bank, ParamId(0), 5.0);
        assert_eq!(value(&bank, ParamId(0)), 5.0, "snaps without a step");
        bank.step();
        assert_eq!(value(&bank, ParamId(0)), 5.0, "stays pinned across steps");
        set_immediate(&mut bank, ParamId(0), 100.0);
        assert_eq!(
            value(&bank, ParamId(0)),
            10.0,
            "clamps to the declared range"
        );
    }

    #[test]
    fn setters_report_unknown_ids_and_non_finite_values() {
        let mut bank = SmootherBank::new(
            &[pinfo(ParamId(0), (0.0, 10.0), 0.0, Smoothing::Linear)],
            &spec(),
        );
        assert!(matches!(
            bank.set_immediate(ParamId(99), 3.0),
            Err(ParamSetError::UnknownParam(ParamId(99)))
        ));
        assert!(matches!(
            bank.set_target(ParamId(99), 3.0),
            Err(ParamSetError::UnknownParam(ParamId(99)))
        ));
        assert!(matches!(
            bank.set_immediate(ParamId(0), f64::NAN),
            Err(ParamSetError::NonFiniteValue {
                param: ParamId(0),
                value
            }) if value.is_nan()
        ));
        assert!(matches!(
            bank.set_target(ParamId(0), f64::INFINITY),
            Err(ParamSetError::NonFiniteValue {
                param: ParamId(0),
                value
            }) if value.is_infinite()
        ));
    }

    /// A `ParamInfo` with an explicit ramp time, for per-param smoothing tests.
    fn pinfo_ramp(
        id: ParamId,
        range: (f64, f64),
        default: f64,
        smoothing: Smoothing,
        smoothing_ms: f64,
    ) -> ParamInfo {
        ParamInfo::new(id, "p", range, default, Unit::Linear)
            .with_smoothing(smoothing)
            .with_smoothing_ms(smoothing_ms)
    }

    #[test]
    fn default_smoothing_is_exponential_for_hz_and_linear_elsewhere() {
        assert_eq!(Smoothing::default_for(Unit::Hz), Smoothing::Exponential);
        for unit in [Unit::Db, Unit::Ms, Unit::Q, Unit::Linear] {
            assert_eq!(Smoothing::default_for(unit), Smoothing::Linear);
        }
    }

    #[test]
    fn exponential_ramps_each_target_at_a_constant_ratio_and_snaps() {
        // 5 ms at 48 kHz is 7.5 control-rate steps. A partial-range target
        // still takes eight updates and uses a ratio derived from the current
        // value and target.
        let info = pinfo_ramp(
            ParamId(0),
            (10.0, 10_000.0),
            10.0,
            Smoothing::Exponential,
            5.0,
        );
        let mut bank = SmootherBank::new(&[info], &spec());
        let steps = (5.0e-3 * 48_000.0) / CR_STEP as f64;
        let r = crate::dsp::math::pow(100.0, 1.0 / steps);

        set_target(&mut bank, ParamId(0), 1_000.0);
        let mut prev = value(&bank, ParamId(0));
        let mut count = 0usize;
        while value(&bank, ParamId(0)) < 1_000.0 {
            bank.step();
            count += 1;
            let cur = value(&bank, ParamId(0));
            assert!(cur > prev, "rising ramp is monotone");
            if cur < 1_000.0 {
                let ratio = cur / prev;
                assert!(
                    (ratio - r).abs() < 1e-12,
                    "constant multiplicative step: {ratio} vs {r}"
                );
            }
            prev = cur;
            assert!(count < 64, "must converge");
        }
        assert_eq!(value(&bank, ParamId(0)), 1_000.0, "snaps exactly to target");
        assert_eq!(count, 8, "partial-range rise completes in ceil(steps)");

        // Falling derives a new ratio for its own target change.
        let down_ratio = crate::dsp::math::pow(10.0, 1.0 / steps);
        set_target(&mut bank, ParamId(0), 100.0);
        bank.step();
        let down = value(&bank, ParamId(0));
        assert!(
            (down - 1_000.0 / down_ratio).abs() / down < 1e-12,
            "falling divides by the ratio: {down}"
        );

        // Reset returns to the default.
        bank.reset();
        assert_eq!(value(&bank, ParamId(0)), 10.0);
    }

    #[test]
    fn linear_ramp_duration_is_per_target_change() {
        // Gain's broad declared range must not turn a moderate target change
        // into a near-instantaneous jump. Repeating the same target also must
        // not restart or slow the active ramp.
        let info = pinfo_ramp(ParamId(0), (-96.0, 24.0), -1.5, Smoothing::Linear, 5.0);
        let mut bank = SmootherBank::new(&[info], &spec());
        let steps = (5.0e-3 * 48_000.0) / CR_STEP as f64;
        let delta = 18.5 / steps;

        set_target(&mut bank, ParamId(0), -20.0);
        bank.step();
        assert!((value(&bank, ParamId(0)) - (-1.5 - delta)).abs() < 1e-12);

        set_target(&mut bank, ParamId(0), -20.0);
        bank.step();
        assert!((value(&bank, ParamId(0)) - (-1.5 - 2.0 * delta)).abs() < 1e-12);

        for _ in 0..6 {
            bank.step();
        }
        assert_eq!(value(&bank, ParamId(0)), -20.0);
    }

    #[test]
    fn per_param_ramp_time_scales_the_traversal() {
        // The same target change with 5 ms and 20 ms Linear ramps. The slower
        // one takes four times as many updates.
        let infos = [
            pinfo_ramp(ParamId(0), (0.0, 1.0), 0.0, Smoothing::Linear, 5.0),
            pinfo_ramp(ParamId(1), (0.0, 1.0), 0.0, Smoothing::Linear, 20.0),
        ];
        let mut bank = SmootherBank::new(&infos, &spec());
        set_target(&mut bank, ParamId(0), 1.0);
        set_target(&mut bank, ParamId(1), 1.0);
        let (mut fast, mut slow) = (0usize, 0usize);
        for _ in 0..256 {
            if value(&bank, ParamId(0)) < 1.0 {
                fast += 1;
            }
            if value(&bank, ParamId(1)) < 1.0 {
                slow += 1;
            }
            bank.step();
        }
        assert_eq!(fast, 8, "5 ms target change takes ceil(7.5) steps");
        assert_eq!(slow, 30, "20 ms target change takes exactly 30 steps");
    }

    #[test]
    fn integer_step_exponential_ramp_finishes_on_time() {
        let info = pinfo_ramp(
            ParamId(0),
            (10.0, 10_000.0),
            10.0,
            Smoothing::Exponential,
            20.0,
        );
        let mut bank = SmootherBank::new(&[info], &spec());
        set_target(&mut bank, ParamId(0), 1_000.0);

        for _ in 0..29 {
            bank.step();
        }
        assert!(value(&bank, ParamId(0)) < 1_000.0);
        bank.step();
        assert_eq!(value(&bank, ParamId(0)), 1_000.0);
    }

    #[test]
    fn validation_rejects_bad_ramps_and_nonpositive_exponential_ranges() {
        // Exponential with a zero range minimum.
        let mut zero_min = KernelProcessor::new(probe::<Probe1Params>(
            vec![pinfo_ramp(
                ParamId(0),
                (0.0, 1.0),
                0.5,
                Smoothing::Exponential,
                5.0,
            )],
            0,
        ));
        assert!(matches!(
            Processor::<f32>::prepare(&mut zero_min, spec()),
            Err(DspError::InvalidParam(_))
        ));
        // Logarithmic value mapping independently requires a positive range.
        let invalid_scale = ParamInfo::new(ParamId(0), "mapped", (0.0, 1.0), 0.5, Unit::Linear)
            .with_value_scale(crate::parameter::ValueScale::Logarithmic);
        let mut invalid_scale = KernelProcessor::new(probe::<Probe1Params>(vec![invalid_scale], 0));
        assert!(matches!(
            Processor::<f32>::prepare(&mut invalid_scale, spec()),
            Err(DspError::InvalidParam(_))
        ));

        // Unit, mapping, and smoothing remain independently overridable.
        let linear_hz = ParamInfo::new(ParamId(0), "hz", (20.0, 20_000.0), 440.0, Unit::Hz)
            .with_value_scale(crate::parameter::ValueScale::Linear)
            .with_smoothing(Smoothing::Linear);
        let mut linear_hz = KernelProcessor::new(probe::<Probe1Params>(vec![linear_hz], 0));
        Processor::<f32>::prepare(&mut linear_hz, spec()).expect("linear Hz is valid");
        // Non-positive and non-finite ramp times.
        for bad_ramp in [0.0, -1.0, f64::NAN] {
            let mut p = KernelProcessor::new(probe::<Probe1Params>(
                vec![pinfo_ramp(
                    ParamId(0),
                    (0.0, 1.0),
                    0.5,
                    Smoothing::Linear,
                    bad_ramp,
                )],
                0,
            ));
            assert!(matches!(
                Processor::<f32>::prepare(&mut p, spec()),
                Err(DspError::InvalidParam(_))
            ));
        }
    }

    // drive split structure.

    crate::params! {
        /// Smoothed parameter values for the recorder test doubles.
        pub struct RecorderParams {
            /// The single recorded parameter.
            pub value => VALUE,
        }
    }

    /// A kernel that records each render's `(start, len, value)`.
    struct Recorder {
        params: Vec<ParamInfo>,
        log: Trace,
    }
    impl Kernel<f32> for Recorder {
        type Params = RecorderParams;
        fn prepare(&mut self, _spec: ProcessSpec) -> Result<(), DspError> {
            Ok(())
        }
        fn reset(&mut self) {}
        fn param_info(&self) -> &[ParamInfo] {
            &self.params
        }
        fn render(&mut self, io: &mut SubBlock<'_, '_, '_, f32>, params: &RecorderParams) {
            self.log.push((io.start, io.len, params.value));
        }
    }

    /// Expected trajectory for an 80-frame block with events at 0 and 40:
    /// grid-aligned runs only; the offset-0 target steps at frame 0 (stream
    /// start is a boundary), and the offset-40 target latches before the step
    /// at 64.
    fn expected_trace() -> Trace {
        vec![(0, 32, 10.0), (32, 32, 10.0), (64, 16, 20.0)]
    }
    fn driver_case() -> (Vec<ParamInfo>, Vec<ParamEvent>) {
        let p = RecorderParams::VALUE;
        (
            vec![pinfo(p, (0.0, 100.0), 0.0, Smoothing::Step)],
            vec![ev(0, p, 10.0), ev(40, p, 20.0)],
        )
    }

    #[test]
    fn drive_splits_on_grid_and_latches_targets() {
        let (params, events) = driver_case();
        let mut rec = Recorder {
            params: params.clone(),
            log: Vec::new(),
        };
        let mut bank = SmootherBank::new(&params, &spec());
        let mut chans: Vec<Vec<f32>> = vec![vec![0.0f32; 80]];
        let mut planes: Vec<&mut [f32]> = chans.iter_mut().map(Vec::as_mut_slice).collect();
        let mut ctx = ProcessContext::in_place(&mut planes, 0).with_events(&events);
        drive(&mut rec, &mut bank, &mut ctx);
        assert_eq!(rec.log, expected_trace());
    }

    #[test]
    fn event_offsets_within_one_grid_cell_share_the_next_boundary() {
        let p = RecorderParams::VALUE;
        let params = vec![pinfo(p, (0.0, 100.0), 0.0, Smoothing::Step)];
        let mut traces = Vec::new();
        for offset in [1, 31] {
            let mut rec = Recorder {
                params: params.clone(),
                log: Vec::new(),
            };
            let mut bank = SmootherBank::new(&params, &spec());
            let mut chans = [vec![0.0f32; 64]];
            let mut planes: Vec<&mut [f32]> = chans.iter_mut().map(Vec::as_mut_slice).collect();
            let events = [ev(offset, p, 10.0)];
            let mut ctx = ProcessContext::in_place(&mut planes, 0).with_events(&events);
            drive(&mut rec, &mut bank, &mut ctx);
            traces.push(rec.log);
        }
        assert_eq!(traces[0], traces[1]);
        assert_eq!(traces[0], vec![(0, 32, 0.0), (32, 32, 10.0)]);
    }

    #[test]
    fn targets_after_the_last_boundary_latch_for_the_next_block() {
        // Block one is 40 frames (boundaries at 0 and 32) with an event at 36,
        // past its last boundary. Block two's step at 64 applies the target.
        let (params, _) = driver_case();
        let p = RecorderParams::VALUE;
        let mut rec = Recorder {
            params: params.clone(),
            log: Vec::new(),
        };
        let mut bank = SmootherBank::new(&params, &spec());
        let mut chans: Vec<Vec<f32>> = vec![vec![0.0f32; 40]];
        {
            let mut planes: Vec<&mut [f32]> = chans.iter_mut().map(Vec::as_mut_slice).collect();
            let events = [ev(36, p, 20.0)];
            let mut ctx = ProcessContext::in_place(&mut planes, 0).with_events(&events);
            drive(&mut rec, &mut bank, &mut ctx);
        }
        assert_eq!(
            rec.log,
            vec![(0, 32, 0.0), (32, 8, 0.0)],
            "block one: no boundary after the event, value unchanged"
        );
        rec.log.clear();
        {
            let mut planes: Vec<&mut [f32]> = chans.iter_mut().map(Vec::as_mut_slice).collect();
            let mut ctx = ProcessContext::in_place(&mut planes, 40);
            drive(&mut rec, &mut bank, &mut ctx);
        }
        assert_eq!(
            rec.log,
            vec![(0, 24, 0.0), (24, 16, 20.0)],
            "block two: the step at 64 applies the carried target"
        );
    }

    /// A kernel that writes the smoothed parameter value into every frame.
    struct WriteValue {
        params: Vec<ParamInfo>,
    }
    impl Kernel<f32> for WriteValue {
        type Params = RecorderParams;
        fn prepare(&mut self, _spec: ProcessSpec) -> Result<(), DspError> {
            Ok(())
        }
        fn reset(&mut self) {}
        fn param_info(&self) -> &[ParamInfo] {
            &self.params
        }
        fn render(&mut self, io: &mut SubBlock<'_, '_, '_, f32>, params: &RecorderParams) {
            let v = params.value as f32;
            for ch in 0..io.channels() {
                io.channel_mut(ch).fill(v);
            }
        }
    }

    #[test]
    fn driven_set_parameter_immediate_pins_the_value_from_frame_zero() {
        let (params, _) = driver_case();
        let mut driven = KernelProcessor::new(WriteValue { params });
        Processor::<f32>::prepare(&mut driven, spec()).unwrap();
        driven
            .set_parameter_immediate(RecorderParams::VALUE, 7.0)
            .expect("known parameter id");
        let mut chans: Vec<Vec<f32>> = vec![vec![0.0f32; 80]];
        let mut planes: Vec<&mut [f32]> = chans.iter_mut().map(Vec::as_mut_slice).collect();
        let mut ctx = ProcessContext::in_place(&mut planes, 0);
        Processor::<f32>::process(&mut driven, &mut ctx);
        assert!(
            chans[0].iter().all(|&x| x == 7.0),
            "exact value from frame 0, no ramp"
        );
    }

    #[test]
    fn driven_set_parameter_immediate_reports_errors_without_changing_the_value() {
        let (params, _) = driver_case();
        let mut driven = KernelProcessor::new(WriteValue { params });
        Processor::<f32>::prepare(&mut driven, spec()).unwrap();
        driven
            .set_parameter_immediate(RecorderParams::VALUE, 4.0)
            .expect("known parameter id");

        assert!(matches!(
            driven.set_parameter_immediate(ParamId(99), 3.0),
            Err(ParamSetError::UnknownParam(ParamId(99)))
        ));
        assert!(matches!(
            driven.set_parameter_immediate(RecorderParams::VALUE, f64::NAN),
            Err(ParamSetError::NonFiniteValue {
                param: RecorderParams::VALUE,
                value
            }) if value.is_nan()
        ));

        let mut chans = [vec![0.0f32; 1]];
        let mut planes: Vec<&mut [f32]> = chans.iter_mut().map(Vec::as_mut_slice).collect();
        let mut ctx = ProcessContext::in_place(&mut planes, 0);
        Processor::<f32>::process(&mut driven, &mut ctx);
        assert_eq!(chans[0][0], 4.0, "rejected writes leave state unchanged");
    }

    #[test]
    fn release_latching_rules_skip_invalid_values_offsets_and_unknown_ids() {
        let params = [pinfo(ParamId(0), (0.0, 10.0), 1.0, Smoothing::Step)];
        let mut bank = SmootherBank::new(&params, &spec());
        let events = [
            ev(0, ParamId(0), f64::NAN),
            ev(0, ParamId(99), 8.0),
            ev(4, ParamId(0), 6.0),
            ev(8, ParamId(0), 9.0),
        ];
        let mut cursor = 0;
        latch_events(&mut bank, &events, &mut cursor, 8, 8);
        bank.step();
        assert_eq!(bank.value(ParamId(0)), Some(6.0));
        assert_eq!(cursor, events.len());
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "parameter events must have offset < frames")]
    fn drive_debug_asserts_out_of_range_events() {
        let (params, _) = driver_case();
        let p = ParamId(0);
        let events = [ev(80, p, 1.0)];
        let mut rec = Recorder {
            params: params.clone(),
            log: Vec::new(),
        };
        let mut bank = SmootherBank::new(&params, &spec());
        let mut chans: Vec<Vec<f32>> = vec![vec![0.0f32; 80]];
        let mut planes: Vec<&mut [f32]> = chans.iter_mut().map(Vec::as_mut_slice).collect();
        let mut ctx = ProcessContext::in_place(&mut planes, 0).with_events(&events);
        drive(&mut rec, &mut bank, &mut ctx);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "parameter events must be sorted by offset")]
    fn drive_debug_asserts_unsorted_events() {
        let (params, _) = driver_case();
        let p = ParamId(0);
        let events = [ev(40, p, 1.0), ev(20, p, 2.0)];
        let mut rec = Recorder {
            params: params.clone(),
            log: Vec::new(),
        };
        let mut bank = SmootherBank::new(&params, &spec());
        let mut chans: Vec<Vec<f32>> = vec![vec![0.0f32; 80]];
        let mut planes: Vec<&mut [f32]> = chans.iter_mut().map(Vec::as_mut_slice).collect();
        let mut ctx = ProcessContext::in_place(&mut planes, 0).with_events(&events);
        drive(&mut rec, &mut bank, &mut ctx);
    }

    // KernelProcessor boundaries.

    crate::params! {
        /// One-field probe parameter set.
        pub struct Probe1Params {
            /// Declaration index 0.
            pub p0 => P0,
        }
    }

    crate::params! {
        /// Two-field probe parameter set.
        pub struct Probe2Params {
            /// Declaration index 0.
            pub p0 => P0,
            /// Declaration index 1.
            pub p1 => P1,
        }
    }

    /// A metadata/footprint probe. Generic over its typed parameter set so
    /// tests can match or deliberately mismatch `Params::COUNT` against the
    /// metadata list.
    struct ProbeKernel<P> {
        params: Vec<ParamInfo>,
        footprint: usize,
        latency: usize,
        sidechain: usize,
        io: IoMode,
        tail: Tail,
        /// The spec the wrapper handed this kernel's `prepare`.
        seen_spec: Option<ProcessSpec>,
        _params: PhantomData<P>,
    }
    fn probe<P>(params: Vec<ParamInfo>, footprint: usize) -> ProbeKernel<P> {
        ProbeKernel {
            params,
            footprint,
            latency: 0,
            sidechain: 0,
            io: IoMode::InPlace,
            tail: Tail::None,
            seen_spec: None,
            _params: PhantomData,
        }
    }
    impl<P: Params> Kernel<f32> for ProbeKernel<P> {
        type Params = P;
        fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
            self.seen_spec = Some(spec);
            Ok(())
        }
        fn reset(&mut self) {}
        fn render(&mut self, _io: &mut SubBlock<'_, '_, '_, f32>, _params: &P) {}
        fn param_info(&self) -> &[ParamInfo] {
            &self.params
        }
        fn memory_footprint(&self) -> usize {
            self.footprint
        }
        fn latency(&self) -> usize {
            self.latency
        }
        fn sidechain_inputs(&self) -> usize {
            self.sidechain
        }
        fn io_mode(&self) -> IoMode {
            self.io
        }
        fn tail(&self) -> Tail {
            self.tail
        }
    }

    fn n_params(n: u32) -> Vec<ParamInfo> {
        (0..n)
            .map(|i| pinfo(ParamId(i), (0.0, 1.0), 0.0, Smoothing::Step))
            .collect()
    }

    #[test]
    fn prepare_rejects_non_sequential_param_ids() {
        // Ids must be sequential from 0 in declaration order. A first id of 1
        // is rejected at prepare.
        let mut kp = KernelProcessor::new(probe::<Probe1Params>(
            vec![pinfo(ParamId(1), (0.0, 1.0), 0.0, Smoothing::Step)],
            0,
        ));
        assert!(matches!(
            Processor::<f32>::prepare(&mut kp, spec()),
            Err(DspError::InvalidParam(_))
        ));
    }

    #[test]
    fn prepare_enforces_the_budget_at_the_boundary() {
        // Total footprint is kernel footprint plus bank footprint.
        let mut measure = KernelProcessor::new(probe::<Probe2Params>(n_params(2), 4096));
        Processor::<f32>::prepare(&mut measure, spec()).unwrap();
        let total = Processor::<f32>::memory_footprint(&measure);

        let mut at = KernelProcessor::new(probe::<Probe2Params>(n_params(2), 4096));
        assert!(
            Processor::<f32>::prepare(&mut at, spec_cap(Some(total))).is_ok(),
            "fits exactly at the cap"
        );
        let mut over = KernelProcessor::new(probe::<Probe2Params>(n_params(2), 4096));
        assert!(
            matches!(
                Processor::<f32>::prepare(&mut over, spec_cap(Some(total - 1))),
                Err(DspError::OverBudget { .. })
            ),
            "one byte over the cap fails"
        );
    }

    #[test]
    fn prepare_hands_the_kernel_the_remaining_budget() {
        // The wrapper reserves the bank's bytes first and hands the kernel
        // only the remainder. The reduced cap is part of the kernel contract.
        let bank_bytes = SmootherBank::footprint_for(2);
        let cap = bank_bytes + 4096;
        let mut kp = KernelProcessor::new(probe::<Probe2Params>(n_params(2), 64));
        Processor::<f32>::prepare(&mut kp, spec_cap(Some(cap))).unwrap();
        assert_eq!(
            kp.kernel.seen_spec.expect("prepared").max_memory,
            Some(cap - bank_bytes),
            "the kernel's sub-budget is the cap minus the bank"
        );

        // Without a cap the spec passes through unchanged.
        let mut kp = KernelProcessor::new(probe::<Probe2Params>(n_params(2), 64));
        Processor::<f32>::prepare(&mut kp, spec()).unwrap();
        assert_eq!(kp.kernel.seen_spec.expect("prepared").max_memory, None);
    }

    #[test]
    fn a_failed_budget_prepare_installs_no_wrapper_state() {
        // The wrapper's own state changes are transactional. Rejecting the cap
        // leaves the smoother bank unbuilt.
        let mut p = KernelProcessor::new(probe::<Probe2Params>(n_params(2), 4096));
        let before = Processor::<f32>::memory_footprint(&p);
        assert!(
            Processor::<f32>::prepare(&mut p, spec_cap(Some(1))).is_err(),
            "a one-byte cap cannot hold the bank"
        );
        assert_eq!(
            Processor::<f32>::memory_footprint(&p),
            before,
            "a failed prepare must not install the bank"
        );
    }

    #[test]
    fn prepare_rejects_a_cap_below_the_bank_alone() {
        // The bank is budgeted before the kernel prepares, so a cap smaller
        // than the bank footprint alone reports the bank as `needed`.
        let bank_bytes = std::mem::size_of::<Smoother>();
        let mut kp = KernelProcessor::new(probe::<Probe1Params>(n_params(1), 4096));
        assert!(matches!(
            Processor::<f32>::prepare(&mut kp, spec_cap(Some(bank_bytes - 1))),
            Err(DspError::OverBudget { needed, cap })
                if needed == bank_bytes && cap == bank_bytes - 1
        ));
    }

    #[test]
    fn driven_delegates_metadata() {
        let k = ProbeKernel::<Probe1Params> {
            params: vec![pinfo(ParamId(0), (0.0, 1.0), 0.0, Smoothing::Step)],
            footprint: 1234,
            latency: 7,
            sidechain: 3,
            io: IoMode::Split,
            tail: Tail::Frames(9),
            seen_spec: None,
            _params: PhantomData,
        };
        let mut kp = KernelProcessor::new(k);
        Processor::<f32>::prepare(&mut kp, spec()).unwrap();
        assert_eq!(Processor::<f32>::param_info(&kp).len(), 1);
        assert_eq!(Processor::<f32>::param_info(&kp)[0].id, ParamId(0));
        assert_eq!(Processor::<f32>::sidechain_inputs(&kp), 3);
        assert_eq!(Processor::<f32>::latency(&kp), 7);
        assert_eq!(Processor::<f32>::io_mode(&kp), IoMode::Split);
        assert_eq!(Processor::<f32>::tail(&kp), Tail::Frames(9));
        // Kernel state plus the one-smoother bank.
        assert_eq!(
            Processor::<f32>::memory_footprint(&kp),
            1234 + std::mem::size_of::<Smoother>()
        );
    }

    #[test]
    fn driven_footprint_saturates_when_the_total_is_not_representable() {
        let mut kp = KernelProcessor::new(probe::<Probe1Params>(n_params(1), usize::MAX));
        Processor::<f32>::prepare(&mut kp, spec()).expect("prepare without a memory cap");
        assert_eq!(Processor::<f32>::memory_footprint(&kp), usize::MAX);
    }

    // Typed-parameter-count validation.

    #[test]
    fn prepare_rejects_more_typed_fields_than_metadata() {
        // Two typed fields, one metadata entry. Without the count check,
        // `Params::from_bank` would panic during render when field 1 reads a
        // smoother the bank never built.
        let mut kp = KernelProcessor::new(probe::<Probe2Params>(n_params(1), 0));
        let before = Processor::<f32>::memory_footprint(&kp);
        assert!(matches!(
            Processor::<f32>::prepare(&mut kp, spec()),
            Err(DspError::InvalidParam(_))
        ));
        assert_eq!(
            Processor::<f32>::memory_footprint(&kp),
            before,
            "a failed prepare must not install smoother bank state"
        );
    }

    #[test]
    fn prepare_rejects_more_metadata_than_typed_fields() {
        // One typed field, two metadata entries. The extra entry would be a
        // phantom automatable parameter the kernel never reads.
        let mut kp = KernelProcessor::new(probe::<Probe1Params>(n_params(2), 0));
        let before = Processor::<f32>::memory_footprint(&kp);
        assert!(matches!(
            Processor::<f32>::prepare(&mut kp, spec()),
            Err(DspError::InvalidParam(_))
        ));
        assert_eq!(
            Processor::<f32>::memory_footprint(&kp),
            before,
            "a failed prepare must not install smoother bank state"
        );
    }

    #[test]
    fn prepare_accepts_matching_count_and_metadata() {
        // The count check does not reject a well-formed kernel.
        let mut kp = KernelProcessor::new(probe::<Probe2Params>(n_params(2), 0));
        assert!(Processor::<f32>::prepare(&mut kp, spec()).is_ok());
    }

    // Debug-time host-geometry validation. Host geometry is a precondition,
    // so each violation panics in debug builds with a focused message.
    #[cfg(debug_assertions)]
    mod geometry {
        use super::*;
        use crate::block::AudioBlock;

        /// A prepared mono in-place processor under the 128-frame test spec.
        fn prepared() -> KernelProcessor<WriteValue> {
            let (params, _) = driver_case();
            let mut p = KernelProcessor::new(WriteValue { params });
            Processor::<f32>::prepare(&mut p, spec()).unwrap();
            p
        }

        fn process_frames(p: &mut KernelProcessor<WriteValue>, channels: usize, frames: usize) {
            let mut chans: Vec<Vec<f32>> = vec![vec![0.0f32; frames]; channels];
            let mut planes: Vec<&mut [f32]> = chans.iter_mut().map(Vec::as_mut_slice).collect();
            let mut ctx = ProcessContext::in_place(&mut planes, 0);
            Processor::<f32>::process(p, &mut ctx);
        }

        fn flush_frames(p: &mut KernelProcessor<WriteValue>, channels: usize, frames: usize) {
            let mut chans: Vec<Vec<f32>> = vec![vec![0.0f32; frames]; channels];
            let mut planes: Vec<&mut [f32]> = chans.iter_mut().map(Vec::as_mut_slice).collect();
            let mut out = AudioBlockMut::new(&mut planes);
            let _ = Processor::<f32>::flush(p, &mut out);
        }

        #[test]
        #[should_panic(expected = "process requires a successful prepare")]
        fn process_before_prepare_panics() {
            let (params, _) = driver_case();
            let mut p = KernelProcessor::new(WriteValue { params });
            process_frames(&mut p, 1, 4);
        }

        #[test]
        #[should_panic(expected = "must not exceed the prepared max_block")]
        fn oversized_block_panics() {
            // spec().max_block is 128.
            let mut p = prepared();
            process_frames(&mut p, 1, 129);
        }

        #[test]
        #[should_panic(expected = "must equal the prepared channel count")]
        fn wrong_channel_count_panics() {
            // Prepared mono, driven stereo.
            let mut p = prepared();
            process_frames(&mut p, 2, 4);
        }

        #[test]
        #[should_panic(expected = "flush output channel count (0)")]
        fn flush_with_too_few_channels_panics() {
            let mut p = prepared();
            flush_frames(&mut p, 0, 4);
        }

        #[test]
        #[should_panic(expected = "flush output channel count (2)")]
        fn flush_with_too_many_channels_panics() {
            let mut p = prepared();
            flush_frames(&mut p, 2, 4);
        }

        #[test]
        fn flush_capacity_may_exceed_process_max_block() {
            let mut p = prepared();
            flush_frames(&mut p, 1, 512);
        }

        #[test]
        #[should_panic(expected = "in-place main frame count (4) must equal ctx.frames (3)")]
        fn in_place_ctx_frames_mismatch_panics() {
            let mut p = prepared();
            let mut chans = [vec![0.0f32; 4]];
            let mut planes: Vec<&mut [f32]> = chans.iter_mut().map(Vec::as_mut_slice).collect();
            let mut ctx = ProcessContext::in_place(&mut planes, 0);
            ctx.frames = 3;
            Processor::<f32>::process(&mut p, &mut ctx);
        }

        #[test]
        #[should_panic(expected = "split input frame count (4) must equal ctx.frames (3)")]
        fn split_ctx_frames_mismatch_panics() {
            let prepared = PreparedContract {
                max_block: 128,
                channels: 1,
                io_mode: IoMode::Split,
                sidechain_inputs: 0,
            };
            let input = [vec![0.0f32; 4]];
            let mut output = [vec![0.0f32; 4]];
            let in_planes: Vec<&[f32]> = input.iter().map(Vec::as_slice).collect();
            let mut out_planes: Vec<&mut [f32]> =
                output.iter_mut().map(Vec::as_mut_slice).collect();
            let mut ctx = ProcessContext::split(&in_planes, &mut out_planes, 0);
            ctx.frames = 3;
            debug_validate_geometry(Some(&prepared), &ctx);
        }

        #[test]
        #[should_panic(expected = "output-only main frame count (4) must equal ctx.frames (3)")]
        fn output_only_ctx_frames_mismatch_panics() {
            let prepared = PreparedContract {
                max_block: 128,
                channels: 1,
                io_mode: IoMode::OutputOnly,
                sidechain_inputs: 0,
            };
            let mut output = [vec![0.0f32; 4]];
            let mut planes: Vec<&mut [f32]> = output.iter_mut().map(Vec::as_mut_slice).collect();
            let mut ctx = ProcessContext::output_only(&mut planes, 0);
            ctx.frames = 3;
            debug_validate_geometry(Some(&prepared), &ctx);
        }

        #[test]
        #[should_panic(expected = "must match the processor's io_mode")]
        fn wrong_io_shape_panics() {
            // An in-place kernel handed a split context.
            let mut p = prepared();
            let input = [vec![0.0f32; 4]];
            let mut output = [vec![0.0f32; 4]];
            let in_planes: Vec<&[f32]> = input.iter().map(Vec::as_slice).collect();
            let mut out_planes: Vec<&mut [f32]> =
                output.iter_mut().map(Vec::as_mut_slice).collect();
            let mut ctx = ProcessContext::split(&in_planes, &mut out_planes, 0);
            Processor::<f32>::process(&mut p, &mut ctx);
        }

        #[test]
        #[should_panic(expected = "must equal sidechain_inputs()")]
        fn sidechain_count_mismatch_panics() {
            // The kernel declares zero sidechain buses; one is supplied.
            let mut p = prepared();
            let key = [vec![0.0f32; 4]];
            let key_planes: Vec<&[f32]> = key.iter().map(Vec::as_slice).collect();
            let sc = [AudioBlock::new(&key_planes)];
            let mut chans = [vec![0.0f32; 4]];
            let mut planes: Vec<&mut [f32]> = chans.iter_mut().map(Vec::as_mut_slice).collect();
            let mut ctx = ProcessContext::in_place(&mut planes, 0).with_sidechains(&sc);
            Processor::<f32>::process(&mut p, &mut ctx);
        }

        #[test]
        #[should_panic(expected = "fewer than the block's")]
        fn short_sidechain_panics() {
            // A declared sidechain bus with fewer frames than the block.
            let mut kernel = probe::<Probe1Params>(n_params(1), 0);
            kernel.sidechain = 1;
            let mut p = KernelProcessor::new(kernel);
            Processor::<f32>::prepare(&mut p, spec()).unwrap();
            let key = [vec![0.0f32; 2]];
            let key_planes: Vec<&[f32]> = key.iter().map(Vec::as_slice).collect();
            let sc = [AudioBlock::new(&key_planes)];
            let mut chans = [vec![0.0f32; 4]];
            let mut planes: Vec<&mut [f32]> = chans.iter_mut().map(Vec::as_mut_slice).collect();
            let mut ctx = ProcessContext::in_place(&mut planes, 0).with_sidechains(&sc);
            Processor::<f32>::process(&mut p, &mut ctx);
        }
    }
}
