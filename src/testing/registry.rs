// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Shared registry of every built-in processor, meter, and rate changer.
//!
//! The registry is the common enrollment point for the cross-cutting suites.
//! Add an entry for each materially different runtime path. A
//! [`ProcessorEntry`], [`MeterEntry`], or [`VariableRateEntry`] enrolls that
//! path in:
//!
//! - the registry-driven contract suite (`tests/registry_contract.rs`):
//!   prepare, metadata consistency, block-size invariance, reset equivalence,
//!   and footprint stability,
//! - the no-allocation suites (`tests/no_alloc.rs` and
//!   `tests/spectral_no_alloc.rs`), which iterate the registry for their armed
//!   processing and flush loops,
//! - the public inventory checks in `tests/documentation.rs`,
//! - listening smoke renders for processor and variable-rate entries.
//!
//! Entries are feature-gated with the same feature as their domain module, so
//! any feature combination compiles. Processor-specific transfer functions,
//! thresholds, spectra, operating modes, and flush semantics stay in the
//! domain test files. The registry only carries what the generic suites need
//! to build and drive an instance.
//!
//! This module is supported test infrastructure for downstream contract suites
//! as well as bisque's repository-wide tests. The entry lists describe the
//! built-ins enabled by the current Cargo feature set.

#![allow(dead_code)]

use core::fmt;

use crate::processor::{
    AudioBlock, AudioBlockMut, DspError, IoMode, Measurer, ProcessSpec, Processor, Produced,
    Source, VariableRate,
};

// ---------------------------------------------------------------------------
// Processor entries
// ---------------------------------------------------------------------------

/// How the shared suites drive an entry's main signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriveMode {
    /// An in-place effect: reads and overwrites the same buffer.
    ///
    /// Declares [`IoMode::InPlace`].
    Effect,
    /// A source that writes output without a main input signal.
    ///
    /// Declares [`IoMode::OutputOnly`].
    Source,
    /// Split I/O: reads a read-only input and writes a disjoint output.
    ///
    /// Declares [`IoMode::Split`].
    Split,
}

impl DriveMode {
    /// The [`IoMode`] an entry with this drive shape must declare.
    #[must_use]
    pub fn io_mode(self) -> IoMode {
        match self {
            DriveMode::Effect => IoMode::InPlace,
            DriveMode::Source => IoMode::OutputOnly,
            DriveMode::Split => IoMode::Split,
        }
    }
}

/// How a registered processor implements the host-facing [`Processor`] contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessorAuthoring {
    /// A [`Kernel`](crate::processor::Kernel) wrapped by `KernelProcessor`.
    Kernel,
    /// A direct [`Processor`] implementation.
    Direct,
}

/// One registered [`Processor`] (a `KernelProcessor` kernel or a direct processor).
#[derive(Clone, Copy, Debug)]
pub struct ProcessorEntry {
    /// The public type name, matching its `docs/src/api-surface.md` row.
    pub name: &'static str,
    /// Unique registry id: the type name plus a variant suffix where one type
    /// registers several configurations (for example `biquad-lowpass`).
    pub id: &'static str,
    /// The Cargo feature that gates the processor's domain module.
    pub feature: &'static str,
    /// The trait used to author the concrete processor.
    pub authoring: ProcessorAuthoring,
    /// Declared main-signal drive shape. Must match the instance's `io_mode()`.
    pub drive: DriveMode,
    /// Declared sidechain bus count. Must match the instance's
    /// `sidechain_inputs()`.
    pub sidechain_inputs: usize,
    /// Build a fresh, unprepared instance.
    pub make: fn() -> Box<dyn Processor<f32> + Send>,
}

/// Return every registered processor entry for the enabled features.
#[must_use]
pub fn processor_entries() -> Vec<ProcessorEntry> {
    // `mut` is unused when no domain feature is enabled.
    #[allow(unused_mut)]
    let mut out = Vec::new();
    #[cfg(feature = "filters")]
    filters_entries(&mut out);
    #[cfg(feature = "dynamics")]
    dynamics_entries(&mut out);
    #[cfg(feature = "mastering")]
    mastering_entries(&mut out);
    #[cfg(feature = "generators")]
    generators_entries(&mut out);
    #[cfg(feature = "time")]
    time_entries(&mut out);
    #[cfg(feature = "repair")]
    repair_entries(&mut out);
    #[cfg(feature = "spectral")]
    spectral_entries(&mut out);
    out
}

#[cfg(feature = "filters")]
fn filters_entries(out: &mut Vec<ProcessorEntry>) {
    use crate::filters::{Biquad, MovingAverage};
    use crate::processor::KernelProcessor;
    let e = |id, make| ProcessorEntry {
        name: "Biquad",
        id,
        feature: "filters",
        authoring: ProcessorAuthoring::Kernel,
        drive: DriveMode::Effect,
        sidechain_inputs: 0,
        make,
    };
    out.push(e("biquad-lowpass", || {
        Box::new(KernelProcessor::new(Biquad::lowpass()))
    }));
    out.push(e("biquad-highpass", || {
        Box::new(KernelProcessor::new(Biquad::highpass()))
    }));
    out.push(e("biquad-low-shelf", || {
        Box::new(KernelProcessor::new(Biquad::low_shelf()))
    }));
    out.push(e("biquad-high-shelf", || {
        Box::new(KernelProcessor::new(Biquad::high_shelf()))
    }));
    out.push(e("biquad-peaking", || {
        Box::new(KernelProcessor::new(Biquad::peaking()))
    }));
    out.push(ProcessorEntry {
        name: "MovingAverage",
        id: "moving-average-16",
        feature: "filters",
        authoring: ProcessorAuthoring::Kernel,
        drive: DriveMode::Split,
        sidechain_inputs: 0,
        make: || Box::new(KernelProcessor::new(MovingAverage::new(16))),
    });
}

#[cfg(feature = "dynamics")]
fn dynamics_entries(out: &mut Vec<ProcessorEntry>) {
    use crate::dynamics::{
        Compressor, CompressorSettings, Expander, ExpanderSettings, Gate, GateSettings,
    };
    use crate::processor::KernelProcessor;
    let e = |name, id, sidechain_inputs, make| ProcessorEntry {
        name,
        id,
        feature: "dynamics",
        authoring: ProcessorAuthoring::Kernel,
        drive: DriveMode::Effect,
        sidechain_inputs,
        make,
    };
    out.push(e("Compressor", "compressor", 0, || {
        Box::new(KernelProcessor::new(Compressor::new()))
    }));
    out.push(e("Compressor", "compressor-sidechain", 1, || {
        Box::new(KernelProcessor::new(Compressor::with_settings(
            CompressorSettings::new().use_sidechain(true),
        )))
    }));
    out.push(e("Expander", "expander", 0, || {
        Box::new(KernelProcessor::new(Expander::new()))
    }));
    out.push(e("Expander", "expander-sidechain", 1, || {
        Box::new(KernelProcessor::new(Expander::with_settings(
            ExpanderSettings::new().use_sidechain(true),
        )))
    }));
    out.push(e("Gate", "gate", 0, || {
        Box::new(KernelProcessor::new(Gate::new()))
    }));
    out.push(e("Gate", "gate-sidechain", 1, || {
        Box::new(KernelProcessor::new(Gate::with_settings(
            GateSettings::new().use_sidechain(true),
        )))
    }));
}

#[cfg(feature = "mastering")]
fn mastering_entries(out: &mut Vec<ProcessorEntry>) {
    use crate::mastering::{Dither, DitherSettings, Gain, Limiter, Scale};
    use crate::processor::KernelProcessor;
    let e = |name, id, make| ProcessorEntry {
        name,
        id,
        feature: "mastering",
        authoring: ProcessorAuthoring::Kernel,
        drive: DriveMode::Effect,
        sidechain_inputs: 0,
        make,
    };
    out.push(e("Gain", "gain", || {
        Box::new(KernelProcessor::new(Gain::new()))
    }));
    // A non-trivial factor so the generic suites exercise a real multiply.
    out.push(e("Scale", "scale", || {
        Box::new(KernelProcessor::new(Scale::from_db(-6.0)))
    }));
    out.push(e("Dither", "dither-16", || {
        // 16-bit at a pinned seed exercises the per-sample RNG and quantizer.
        Box::new(KernelProcessor::new(Dither::with_settings(
            DitherSettings::new().bits(16).seed(0x5EED_0001),
        )))
    }));
    out.push(e("Limiter", "limiter", || {
        Box::new(KernelProcessor::new(Limiter::new()))
    }));
}

#[cfg(feature = "generators")]
fn generators_entries(out: &mut Vec<ProcessorEntry>) {
    use crate::generators::{PolyBlepOsc, SineOsc, WhiteNoise};
    use crate::processor::KernelProcessor;
    let e = |name, id, make| ProcessorEntry {
        name,
        id,
        feature: "generators",
        authoring: ProcessorAuthoring::Kernel,
        drive: DriveMode::Source,
        sidechain_inputs: 0,
        make,
    };
    out.push(e("SineOsc", "sine-osc", || {
        Box::new(KernelProcessor::new(SineOsc::new()))
    }));
    out.push(e("WhiteNoise", "white-noise", || {
        Box::new(KernelProcessor::new(WhiteNoise::new()))
    }));
    out.push(e("PolyBlepOsc", "poly-blep-saw", || {
        Box::new(KernelProcessor::new(PolyBlepOsc::saw()))
    }));
    out.push(e("PolyBlepOsc", "poly-blep-square", || {
        Box::new(KernelProcessor::new(PolyBlepOsc::square()))
    }));
}

#[cfg(feature = "time")]
fn time_entries(out: &mut Vec<ProcessorEntry>) {
    use crate::processor::KernelProcessor;
    use crate::time::Delay;
    out.push(ProcessorEntry {
        name: "Delay",
        id: "delay",
        feature: "time",
        authoring: ProcessorAuthoring::Kernel,
        drive: DriveMode::Effect,
        sidechain_inputs: 0,
        make: || Box::new(KernelProcessor::new(Delay::new())),
    });
}

#[cfg(feature = "repair")]
fn repair_entries(out: &mut Vec<ProcessorEntry>) {
    use crate::processor::KernelProcessor;
    use crate::repair::{DcBlocker, DcOffset};
    out.push(ProcessorEntry {
        name: "DcBlocker",
        id: "dc-blocker",
        feature: "repair",
        authoring: ProcessorAuthoring::Kernel,
        drive: DriveMode::Effect,
        sidechain_inputs: 0,
        make: || Box::new(KernelProcessor::new(DcBlocker::new())),
    });
    out.push(ProcessorEntry {
        name: "DcOffset",
        id: "dc-offset",
        feature: "repair",
        authoring: ProcessorAuthoring::Kernel,
        drive: DriveMode::Effect,
        sidechain_inputs: 0,
        make: || Box::new(KernelProcessor::new(DcOffset::broadcast(0.1))),
    });
}

#[cfg(feature = "spectral")]
fn spectral_entries(out: &mut Vec<ProcessorEntry>) {
    use crate::spectral::SpectralFilter;
    out.push(ProcessorEntry {
        name: "SpectralFilter",
        id: "spectral-filter-lp",
        feature: "spectral",
        authoring: ProcessorAuthoring::Direct,
        drive: DriveMode::Split,
        sidechain_inputs: 0,
        make: || Box::new(SpectralFilter::low_pass(1024, 512, 6_000.0)),
    });
}

// ---------------------------------------------------------------------------
// Meter entries
// ---------------------------------------------------------------------------

/// A [`Measurer`] with its associated `Reading` type erased.
///
/// `reading_debug` returns the reading's `Debug` rendering, which for the
/// built-in `f64`, `u64`, and struct readings is a round-trip-exact
/// representation, so string equality is reading equality.
pub trait AnyMeter: Send {
    /// See [`Measurer::prepare`].
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError>;
    /// See [`Measurer::reset`].
    fn reset(&mut self);
    /// See [`Measurer::latency`].
    fn latency(&self) -> usize;
    /// See [`Measurer::memory_footprint`].
    fn memory_footprint(&self) -> usize;
    /// See [`Measurer::observe`].
    fn observe(&mut self, block: AudioBlock<'_, '_, f32>);
    /// The current reading's `Debug` rendering.
    fn reading_debug(&self) -> String;
}

impl<M> AnyMeter for M
where
    M: Measurer<f32> + Send,
    M::Reading: fmt::Debug,
{
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        Measurer::<f32>::prepare(self, spec)
    }
    fn reset(&mut self) {
        Measurer::<f32>::reset(self);
    }
    fn latency(&self) -> usize {
        Measurer::<f32>::latency(self)
    }
    fn memory_footprint(&self) -> usize {
        Measurer::<f32>::memory_footprint(self)
    }
    fn observe(&mut self, block: AudioBlock<'_, '_, f32>) {
        Measurer::<f32>::observe(self, block);
    }
    fn reading_debug(&self) -> String {
        format!("{:?}", Measurer::<f32>::read(self))
    }
}

/// One registered [`Measurer`].
#[derive(Clone, Copy, Debug)]
pub struct MeterEntry {
    /// The public type name, matching its `docs/src/api-surface.md` row.
    pub name: &'static str,
    /// Unique registry id.
    pub id: &'static str,
    /// The Cargo feature that gates the meter's domain module.
    pub feature: &'static str,
    /// Build a fresh, unprepared instance.
    pub make: fn() -> Box<dyn AnyMeter>,
}

/// Return every registered meter entry for the enabled features.
#[must_use]
pub fn meter_entries() -> Vec<MeterEntry> {
    // `mut` is unused when `analysis` is disabled.
    #[allow(unused_mut)]
    let mut out = Vec::new();
    #[cfg(feature = "analysis")]
    analysis_entries(&mut out);
    out
}

#[cfg(feature = "analysis")]
fn analysis_entries(out: &mut Vec<MeterEntry>) {
    use crate::analysis::{
        ClipMeter, CrestMeter, LoudnessMeter, LoudnessMeterSettings, MeanMeter, PeakMeter,
        RmsMeter, TruePeakMeter, WindowedRmsMeter, WindowedRmsMeterSettings,
    };
    let e = |name, id, make| MeterEntry {
        name,
        id,
        feature: "analysis",
        make,
    };
    out.push(e("PeakMeter", "peak-meter", || Box::new(PeakMeter::new())));
    out.push(e("RmsMeter", "rms-meter", || Box::new(RmsMeter::new())));
    out.push(e("MeanMeter", "mean-meter", || Box::new(MeanMeter::new())));
    out.push(e("CrestMeter", "crest-meter", || {
        Box::new(CrestMeter::new())
    }));
    out.push(e("TruePeakMeter", "true-peak-meter", || {
        Box::new(TruePeakMeter::new())
    }));
    out.push(e("WindowedRmsMeter", "windowed-rms-meter-512", || {
        Box::new(WindowedRmsMeter::with_settings(
            WindowedRmsMeterSettings::new().window_frames(512),
        ))
    }));
    out.push(e("LoudnessMeter", "loudness-meter", || {
        Box::new(LoudnessMeter::with_settings(
            LoudnessMeterSettings::with_max_integrated_seconds(2.0),
        ))
    }));
    out.push(e("ClipMeter", "clip-meter", || Box::new(ClipMeter::new())));
}

// ---------------------------------------------------------------------------
// Variable-rate entries
// ---------------------------------------------------------------------------

/// One registered [`VariableRate`] processor.
#[derive(Clone, Copy, Debug)]
pub struct VariableRateEntry {
    /// The public type name, matching its `docs/src/api-surface.md` row.
    pub name: &'static str,
    /// Unique registry id.
    pub id: &'static str,
    /// The Cargo feature that gates the rate changer's domain module.
    pub feature: &'static str,
    /// Build a fresh, unprepared instance.
    pub make: fn() -> Box<dyn VariableRate<f32> + Send>,
}

/// Return every registered variable-rate entry for the enabled features.
#[must_use]
pub fn variable_rate_entries() -> Vec<VariableRateEntry> {
    // `mut` is unused when `time` is disabled.
    #[allow(unused_mut)]
    let mut out = Vec::new();
    #[cfg(feature = "time")]
    time_vr_entries(&mut out);
    out
}

#[cfg(feature = "time")]
fn time_vr_entries(out: &mut Vec<VariableRateEntry>) {
    use crate::time::{TimeStretch, TimeStretchSettings};
    out.push(VariableRateEntry {
        name: "TimeStretch",
        id: "time-stretch-1.5x",
        feature: "time",
        make: || {
            Box::new(TimeStretch::<f32>::with_settings(
                TimeStretchSettings::new().stretch(1.5),
            ))
        },
    });
}

// ---------------------------------------------------------------------------
// Boxed variable-rate adapter
// ---------------------------------------------------------------------------

/// A boxed registry rate changer as a sized [`VariableRate`].
pub struct BoxedVariableRate(pub Box<dyn VariableRate<f32> + Send>);

impl fmt::Debug for BoxedVariableRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("BoxedVariableRate(..)")
    }
}

impl VariableRate<f32> for BoxedVariableRate {
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        self.0.prepare(spec)
    }
    fn reset(&mut self) {
        self.0.reset();
    }
    fn latency(&self) -> usize {
        self.0.latency()
    }
    fn memory_footprint(&self) -> usize {
        self.0.memory_footprint()
    }
    fn process(
        &mut self,
        input: &mut dyn Source<f32>,
        out: &mut AudioBlockMut<'_, '_, f32>,
    ) -> Produced {
        self.0.process(input, out)
    }
}
