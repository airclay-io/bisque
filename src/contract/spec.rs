// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! The fixed processing configuration, [`ProcessSpec`].

/// The fixed spec a processor is prepared for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcessSpec {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel count of the main signal.
    pub channels: usize,
    /// The largest block length supplied to same-rate processing or metering.
    pub max_block: usize,
    /// Optional cap on internal state, measured in logical reserved payload
    /// bytes: every processor-owned element slot intentionally kept available
    /// after prepare, whether or not it currently contains valid history,
    /// times element size. This is the same measure `memory_footprint()`
    /// reports. Incidental allocator overcapacity, inline scalar state,
    /// container metadata, and allocator bookkeeping are outside the budget,
    /// so the cap bounds the DSP state model, not the process's heap usage. When
    /// a third-party backend owns opaque plan storage that it does not expose,
    /// that storage is also outside the logical measure; caller-owned scratch
    /// buffers remain included. If the processor cannot fit, it returns
    /// [`DspError::OverBudget`](crate::processor::DspError::OverBudget) from `prepare`.
    ///
    /// Built-ins check the cap before committing state when the layout is
    /// known. Failed prepares leave the processor unprepared. A downstream
    /// kernel that ignores its sub-budget may still allocate before the
    /// wrapper rejects the total (see
    /// [`KernelProcessor`](crate::processor::KernelProcessor)).
    pub max_memory: Option<usize>,
}
