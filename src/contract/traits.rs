// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Traits for processors, kernels, meters, rate changers, and sources.
//!
//! Most same-rate processors implement [`Kernel`] and use
//! [`Kernel::into_processor`] to obtain a [`Processor`]. Sources are kernels
//! that ignore buffer contents and overwrite the output. [`VariableRate`] is
//! a pull/produce contract with no parameter automation; configuration is
//! fixed at construction.

use crate::block::{AudioBlock, AudioBlockMut, IoMode};
use crate::context::{ProcessContext, Produced, SubBlock, Tail};
use crate::error::DspError;
use crate::param::{ParamId, ParamInfo, ParamSetError, Params};
use crate::processor::KernelProcessor;
use crate::processor::Sample;
use crate::spec::ProcessSpec;

/// A same-rate transform driven by a host. Most processors are [`Kernel`]s
/// wrapped in [`KernelProcessor`]. Implement `Processor` directly for block-based
/// processors such as FFT processors with latency.
///
/// The trait itself does not require `Send` or `Sync`. Bisque's built-in
/// processors are `Send`; `Sync` is not promised because processing mutates
/// state through `&mut self`.
pub trait Processor<T: Sample> {
    /// Allocate and configure for `spec`.
    ///
    /// # Errors
    /// Invalid parameters, an unsupported spec, or an over-tight memory budget.
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError>;
    /// Return to the post-`prepare` state.
    fn reset(&mut self);
    /// Processing latency in frames, constant after `prepare`.
    fn latency(&self) -> usize {
        0
    }
    /// Output that continues after input ends.
    fn tail(&self) -> Tail {
        Tail::None
    }
    /// I/O shape the host provides for the main signal.
    fn io_mode(&self) -> IoMode {
        IoMode::InPlace
    }
    /// Internal state in logical reserved payload bytes, valid after a
    /// successful `prepare`: every processor-owned element slot intentionally
    /// kept available, whether or not it currently contains valid history,
    /// times element size. Incidental allocator overcapacity, inline scalar
    /// state, container metadata, and allocator bookkeeping are excluded.
    /// [`ProcessSpec::max_memory`] caps this same measure.
    fn memory_footprint(&self) -> usize {
        0
    }
    /// The automatable parameters, borrowed from `&self`.
    fn param_info(&self) -> &[ParamInfo] {
        &[]
    }
    /// Number of key/sidechain input buses expected in `ctx.sidechain`.
    fn sidechain_inputs(&self) -> usize {
        0
    }
    /// Snap a declared parameter's current and target values without a ramp.
    ///
    /// This object-safe method lets generic hosts restore state through a trait
    /// object without downcasting. Parameterless processors inherit the
    /// `UnknownParam` default. Process-time [`ParamEvent`](crate::parameter::ParamEvent)
    /// handling remains best-effort because [`Self::process`] returns `()`.
    fn set_parameter_immediate(&mut self, id: ParamId, value: f64) -> Result<(), ParamSetError> {
        let _ = value;
        Err(ParamSetError::UnknownParam(id))
    }
    /// Process one block, including its parameter events.
    ///
    /// Built-in processors treat non-finite audio samples as silence at their
    /// input boundaries. Hosts must still provide the prepared channel count,
    /// block size, sidechains, and I/O shape.
    fn process(&mut self, ctx: &mut ProcessContext<'_, '_, T>);
    /// Drain latency or tail into `out`.
    ///
    /// Writes at most `out.frames()` frames per call and returns how many were
    /// written plus whether the drain has delivered the declared tail
    /// (`done`). Any total cap on a drain is the host's: stop calling when
    /// enough frames have arrived. A [`Tail::Infinite`] processor never
    /// reports `done`, so its host must cap the drain itself. New input
    /// (`process`) or `reset` starts a new drain. After a call reports `done`,
    /// later flush calls write zero frames and remain done until a new drain
    /// starts. `out` must have the prepared main-channel count. Its per-call
    /// frame capacity may exceed `ProcessSpec::max_block`, which constrains
    /// `process` input only.
    fn flush(&mut self, out: &mut AudioBlockMut<'_, '_, T>) -> Produced {
        let _ = out;
        Produced {
            frames: 0,
            done: true,
        }
    }
}

impl<T, P> Processor<T> for Box<P>
where
    T: Sample,
    P: Processor<T> + ?Sized,
{
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        (**self).prepare(spec)
    }

    fn reset(&mut self) {
        (**self).reset();
    }

    fn latency(&self) -> usize {
        (**self).latency()
    }

    fn tail(&self) -> Tail {
        (**self).tail()
    }

    fn io_mode(&self) -> IoMode {
        (**self).io_mode()
    }

    fn memory_footprint(&self) -> usize {
        (**self).memory_footprint()
    }

    fn param_info(&self) -> &[ParamInfo] {
        (**self).param_info()
    }

    fn sidechain_inputs(&self) -> usize {
        (**self).sidechain_inputs()
    }

    fn set_parameter_immediate(&mut self, id: ParamId, value: f64) -> Result<(), ParamSetError> {
        (**self).set_parameter_immediate(id, value)
    }

    fn process(&mut self, ctx: &mut ProcessContext<'_, '_, T>) {
        (**self).process(ctx);
    }

    fn flush(&mut self, out: &mut AudioBlockMut<'_, '_, T>) -> Produced {
        (**self).flush(out)
    }
}

/// A fixed-parameter renderer for a contiguous frame range. Wrap it into a
/// [`Processor`] with [`into_processor`](Kernel::into_processor).
pub trait Kernel<T: Sample> {
    /// The typed parameter set `render` reads, declared with
    /// [`params!`](crate::params). Use [`NoParams`](crate::parameter::NoParams) when the
    /// kernel has no automatable parameters.
    type Params: Params;

    /// Allocate and configure for `spec`.
    ///
    /// # Errors
    /// Invalid parameters or an unsupported spec.
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError>;
    /// Return to the post-`prepare` state.
    fn reset(&mut self);
    /// Processing latency in frames, constant after `prepare`.
    fn latency(&self) -> usize {
        0
    }
    /// Output that continues after input ends.
    fn tail(&self) -> Tail {
        Tail::None
    }
    /// I/O shape the host provides.
    fn io_mode(&self) -> IoMode {
        IoMode::InPlace
    }
    /// The kernel's own state in logical reserved payload bytes: intentionally
    /// available element slots times element size. Incidental allocator
    /// overcapacity, inline scalars, container metadata, and allocator
    /// bookkeeping are excluded.
    ///
    /// The framework-owned smoother bank is sized and added by
    /// [`KernelProcessor`]. It is not counted here.
    fn memory_footprint(&self) -> usize {
        0
    }
    /// The automatable parameters, borrowed from `&self`.
    ///
    /// Lists [`Self::Params`](Kernel::Params)'s fields in declaration order
    /// (sequential ids from `0`). `prepare` validates the ordering. The list is
    /// stable from construction because [`KernelProcessor`] reads it before
    /// `prepare`.
    fn param_info(&self) -> &[ParamInfo] {
        &[]
    }
    /// Number of key/sidechain input buses expected.
    fn sidechain_inputs(&self) -> usize {
        0
    }

    /// Wrap as a [`Processor`] that owns the framework smoother bank.
    ///
    /// The returned [`KernelProcessor`] carries this trait invocation's sample
    /// type, so `<K as Kernel<f64>>::into_processor(k)` produces a
    /// `KernelProcessor<K, f64>`.
    fn into_processor(self) -> KernelProcessor<Self, T>
    where
        Self: Sized,
    {
        KernelProcessor::with_sample_type(self)
    }

    /// Render one sub-block. `params` holds the smoothed values, constant for
    /// the whole sub-block.
    ///
    /// A sub-block spans at most one control-rate cell (32 frames).
    /// [`KernelProcessor`] splits blocks at the grid, and `SubBlock`
    /// construction is crate-internal. Kernels may size per-run scratch to that
    /// bound.
    ///
    /// Built-in kernels treat non-finite audio samples as silence at their input
    /// boundaries.
    fn render(&mut self, io: &mut SubBlock<'_, '_, '_, T>, params: &Self::Params);

    /// Drain latency or tail into `out`, at most `out.frames()` frames per
    /// call. Any total cap is the host's; see
    /// [`Processor::flush`](crate::processor::Processor::flush). The wrapped
    /// processor validates that `out` has the prepared channel count; its frame
    /// capacity is independent of `ProcessSpec::max_block`.
    fn flush(&mut self, out: &mut AudioBlockMut<'_, '_, T>) -> Produced {
        let _ = out;
        Produced {
            frames: 0,
            done: true,
        }
    }
}

/// An analyzer or meter that observes a signal and produces a measurement.
pub trait Measurer<T: Sample> {
    /// A copyable readout such as a peak in dBFS, an RMS, a spectrum slice, a
    /// measured DC offset or normalization gain).
    type Reading;
    /// Allocate and configure for `spec` (window lengths, FFT buffers, a
    /// true-peak oversampler). A successful prepare is required before
    /// [`observe`](Self::observe).
    ///
    /// # Errors
    /// An unsupported spec.
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError>;
    /// Clear accumulated state for reuse.
    fn reset(&mut self);
    /// Detector or window group delay in input frames, constant after
    /// `prepare`.
    ///
    /// A nonzero latency means the reading describes the input roughly this
    /// many frames before the most recently observed one.
    fn latency(&self) -> usize {
        0
    }
    /// Internal state in logical reserved payload bytes, valid after a
    /// successful `prepare` (the same measure as
    /// [`Processor::memory_footprint`]).
    fn memory_footprint(&self) -> usize {
        0
    }
    /// Accumulate over a block whose channel count matches the prepared spec
    /// and whose frame count does not exceed `ProcessSpec::max_block`.
    ///
    /// Geometry is a host precondition. Built-in meters report focused debug
    /// failures for violations and treat non-finite audio samples as silence.
    fn observe(&mut self, block: AudioBlock<'_, '_, T>);
    /// The current measurement.
    fn read(&self) -> Self::Reading;
}

/// A rate changer such as time-stretch or pitch-shift. Input and output frame
/// counts may differ, so it is not a [`Processor`].
///
/// The contract is pull/produce only: configuration is fixed at construction
/// and there is no parameter automation. Output-timeline automation would
/// need a concrete mapping between event time and output frames plus a shared
/// driver, and is deliberately not promised here until an implementation
/// requires it.
pub trait VariableRate<T: Sample> {
    /// Allocate and configure for `spec`.
    ///
    /// # Errors
    /// Invalid parameters or an unsupported spec.
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError>;
    /// Return to the post-`prepare` state.
    fn reset(&mut self);
    /// Processing latency in frames.
    fn latency(&self) -> usize {
        0
    }
    /// Internal state in logical reserved payload bytes, valid after a
    /// successful `prepare` (the same measure as
    /// [`Processor::memory_footprint`]).
    fn memory_footprint(&self) -> usize {
        0
    }
    /// Pull from `input` and write up to `out.frames()` frames. `done` means
    /// input ended and internal buffers drained.
    fn process(
        &mut self,
        input: &mut dyn Source<T>,
        out: &mut AudioBlockMut<'_, '_, T>,
    ) -> Produced;
}

/// Pull input for a [`VariableRate`]. Fewer frames with `done == false` is an
/// underrun. `done == true` is end of input.
pub trait Source<T: Sample> {
    /// Channel count, matching the consumer's prepared input geometry.
    fn channels(&self) -> usize;
    /// Pull up to `out.frames()` input frames into `out`.
    fn pull(&mut self, out: &mut AudioBlockMut<'_, '_, T>) -> Produced;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Exercises default trait methods.
    struct Defaults;
    impl Processor<f32> for Defaults {
        fn prepare(&mut self, _spec: ProcessSpec) -> Result<(), DspError> {
            Ok(())
        }
        fn reset(&mut self) {}
        fn process(&mut self, _ctx: &mut ProcessContext<'_, '_, f32>) {}
    }
    impl Kernel<f32> for Defaults {
        type Params = crate::parameter::NoParams;
        fn prepare(&mut self, _spec: ProcessSpec) -> Result<(), DspError> {
            Ok(())
        }
        fn reset(&mut self) {}
        fn render(
            &mut self,
            _io: &mut SubBlock<'_, '_, '_, f32>,
            _params: &crate::parameter::NoParams,
        ) {
        }
    }
    impl VariableRate<f32> for Defaults {
        fn prepare(&mut self, _spec: ProcessSpec) -> Result<(), DspError> {
            Ok(())
        }
        fn reset(&mut self) {}
        fn process(
            &mut self,
            _input: &mut dyn Source<f32>,
            _out: &mut AudioBlockMut<'_, '_, f32>,
        ) -> Produced {
            Produced {
                frames: 0,
                done: true,
            }
        }
    }
    impl Measurer<f32> for Defaults {
        type Reading = ();
        fn prepare(&mut self, _spec: ProcessSpec) -> Result<(), DspError> {
            Ok(())
        }
        fn reset(&mut self) {}
        fn observe(&mut self, _block: AudioBlock<'_, '_, f32>) {}
        fn read(&self) -> Self::Reading {}
    }

    #[test]
    fn processor_defaults_are_zero_empty_inplace() {
        let mut d = Defaults;
        assert_eq!(Processor::<f32>::latency(&d), 0);
        assert_eq!(Processor::<f32>::memory_footprint(&d), 0);
        assert_eq!(Processor::<f32>::sidechain_inputs(&d), 0);
        assert_eq!(Processor::<f32>::tail(&d), Tail::None);
        assert_eq!(Processor::<f32>::io_mode(&d), IoMode::InPlace);
        assert!(Processor::<f32>::param_info(&d).is_empty());
        assert!(matches!(
            Processor::<f32>::set_parameter_immediate(&mut d, ParamId(9), 1.0),
            Err(ParamSetError::UnknownParam(ParamId(9)))
        ));
    }

    #[test]
    fn kernel_defaults_are_zero_empty_inplace() {
        let d = Defaults;
        assert_eq!(Kernel::<f32>::latency(&d), 0);
        assert_eq!(Kernel::<f32>::memory_footprint(&d), 0);
        assert_eq!(Kernel::<f32>::sidechain_inputs(&d), 0);
        assert_eq!(Kernel::<f32>::tail(&d), Tail::None);
        assert_eq!(Kernel::<f32>::io_mode(&d), IoMode::InPlace);
        assert!(Kernel::<f32>::param_info(&d).is_empty());
    }

    #[test]
    fn variable_rate_defaults() {
        let d = Defaults;
        assert_eq!(VariableRate::<f32>::latency(&d), 0);
        assert_eq!(VariableRate::<f32>::memory_footprint(&d), 0);
    }

    #[test]
    fn measurer_defaults_are_zero() {
        let d = Defaults;
        assert_eq!(Measurer::<f32>::latency(&d), 0);
        assert_eq!(Measurer::<f32>::memory_footprint(&d), 0);
    }

    #[test]
    fn processor_flush_default_yields_nothing_and_is_done() {
        let mut d = Defaults;
        let mut chans: Vec<Vec<f32>> = vec![vec![0.0f32; 4]];
        let p = {
            let mut planes: Vec<&mut [f32]> = chans.iter_mut().map(Vec::as_mut_slice).collect();
            let mut block = AudioBlockMut::new(&mut planes);
            Processor::<f32>::flush(&mut d, &mut block)
        };
        assert_eq!(p.frames, 0);
        assert!(p.done);
    }
}
