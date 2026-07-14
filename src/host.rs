// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Optional lifecycle support for hosting one processor.

use std::marker::PhantomData;

use crate::dsp::driver::{
    debug_validate_flush_geometry, debug_validate_geometry, PreparedContract,
};
use crate::parameter::{ParamEvent, ParamId, ParamInfo, ParamSetError};
use crate::processor::{
    AudioBlock, AudioBlockMut, DspError, Io, IoMode, Kernel, KernelProcessor, ProcessContext,
    ProcessSpec, Processor, Produced, Sample, Tail,
};

/// A successfully prepared owner around one [`Processor`].
///
/// This helper owns universal single-processor lifecycle mechanics: the
/// prepared spec, absolute processing timeline, geometry validation, reset/seek,
/// direct parameter restoration, and flush delegation. It intentionally owns
/// no graph, queue, transport, file, allocation, or tail-length policy.
#[derive(Debug)]
pub struct PreparedProcessor<P, T: Sample = f32> {
    processor: P,
    spec: ProcessSpec,
    sample_pos: u64,
    prepared: PreparedContract,
    _sample: PhantomData<fn() -> T>,
}

impl<P, T> PreparedProcessor<P, T>
where
    T: Sample,
    P: Processor<T>,
{
    /// Prepare an existing processor exactly once and take ownership of it.
    pub fn prepare(mut processor: P, spec: ProcessSpec) -> Result<Self, DspError> {
        processor.prepare(spec)?;
        let prepared = PreparedContract {
            max_block: spec.max_block,
            channels: spec.channels,
            io_mode: processor.io_mode(),
            sidechain_inputs: processor.sidechain_inputs(),
        };
        Ok(Self {
            processor,
            spec,
            sample_pos: 0,
            prepared,
            _sample: PhantomData,
        })
    }

    /// The successfully prepared process specification.
    #[must_use]
    pub const fn spec(&self) -> &ProcessSpec {
        &self.spec
    }

    /// The absolute position of the next processed frame.
    #[must_use]
    pub const fn sample_pos(&self) -> u64 {
        self.sample_pos
    }

    /// Process one block with explicit main I/O, sidechains, and events.
    pub fn process<'view, 'samples>(
        &mut self,
        main: Io<'view, 'samples, T>,
        sidechains: &'view [AudioBlock<'view, 'samples, T>],
        events: &'view [ParamEvent],
    ) {
        let mut ctx = ProcessContext::from_io(main, self.sample_pos)
            .with_sidechains(sidechains)
            .with_events(events);
        debug_validate_geometry(Some(&self.prepared), &ctx);
        let frames = ctx.frames;
        self.processor.process(&mut ctx);
        if frames != 0 {
            self.sample_pos = self.sample_pos.saturating_add(frames as u64);
        }
    }

    /// Process an in-place block without sidechains.
    pub fn process_in_place<'view>(
        &mut self,
        planes: &'view mut [&mut [T]],
        events: &'view [ParamEvent],
    ) {
        self.process(Io::InPlace(AudioBlockMut::new(planes)), &[], events);
    }

    /// Process an output-only block without sidechains.
    pub fn process_output_only<'view>(
        &mut self,
        output: &'view mut [&mut [T]],
        events: &'view [ParamEvent],
    ) {
        self.process(Io::OutputOnly(AudioBlockMut::new(output)), &[], events);
    }

    /// Process a split-I/O block without sidechains.
    pub fn process_split<'view, 'samples>(
        &mut self,
        input: &'view [&'samples [T]],
        output: &'view mut [&'samples mut [T]],
        events: &'view [ParamEvent],
    ) {
        self.process(
            Io::Split {
                input: AudioBlock::new(input),
                output: AudioBlockMut::new(output),
            },
            &[],
            events,
        );
    }

    /// Reset processor state and restart the processing timeline at zero.
    pub fn restart(&mut self) {
        self.processor.reset();
        self.sample_pos = 0;
    }

    /// Reset processor state and place the next block at `sample_pos`.
    pub fn seek(&mut self, sample_pos: u64) {
        self.processor.reset();
        self.sample_pos = sample_pos;
    }

    /// Snap one declared parameter without a smoothing ramp.
    pub fn set_parameter_immediate(
        &mut self,
        id: ParamId,
        value: f64,
    ) -> Result<(), ParamSetError> {
        self.processor.set_parameter_immediate(id, value)
    }

    /// Processing latency in frames.
    #[must_use]
    pub fn latency(&self) -> usize {
        self.processor.latency()
    }

    /// Output remaining after input ends.
    #[must_use]
    pub fn tail(&self) -> Tail {
        self.processor.tail()
    }

    /// Prepared main I/O shape.
    #[must_use]
    pub fn io_mode(&self) -> IoMode {
        self.processor.io_mode()
    }

    /// Number of required sidechain buses.
    #[must_use]
    pub fn sidechain_inputs(&self) -> usize {
        self.processor.sidechain_inputs()
    }

    /// Declared automatable parameters.
    #[must_use]
    pub fn param_info(&self) -> &[ParamInfo] {
        self.processor.param_info()
    }

    /// Logical reserved processor-owned payload bytes.
    #[must_use]
    pub fn memory_footprint(&self) -> usize {
        self.processor.memory_footprint()
    }

    /// Drain processor-local latency or tail without advancing input time.
    pub fn flush(&mut self, out: &mut AudioBlockMut<'_, '_, T>) -> Produced {
        debug_validate_flush_geometry(Some(&self.prepared), out);
        self.processor.flush(out)
    }

    /// Immutable access for processor-specific readouts.
    #[must_use]
    pub const fn processor(&self) -> &P {
        &self.processor
    }

    /// Consume the wrapper and return the processor for direct host control.
    #[must_use]
    pub fn into_inner(self) -> P {
        self.processor
    }
}

impl<K> PreparedProcessor<KernelProcessor<K, f32>, f32>
where
    K: Kernel<f32>,
{
    /// Wrap and prepare a kernel for the ergonomic default `f32` path.
    pub fn prepare_kernel(kernel: K, spec: ProcessSpec) -> Result<Self, DspError> {
        Self::prepare(KernelProcessor::new(kernel), spec)
    }
}

impl<K, T> PreparedProcessor<KernelProcessor<K, T>, T>
where
    T: Sample,
    K: Kernel<T>,
{
    /// Wrap and prepare a kernel for an explicit sample type such as `f64`.
    pub fn prepare_kernel_with_sample_type(kernel: K, spec: ProcessSpec) -> Result<Self, DspError> {
        Self::prepare(KernelProcessor::with_sample_type(kernel), spec)
    }
}
